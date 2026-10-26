# processor/import_processor/nodes/node_pdf_to_md.py
import json
import logging
import shutil
import time
import zipfile

from pathlib import Path

import requests

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ConfigurationError, \
    PdfConversionError
from processor.import_processor.state import ImportGraphState
from utils.mineru_pending_utils import record_batch, update_state, STATE_DONE, STATE_FAILED, STATE_PENDING, \
    STATE_TIMEOUT



class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):

        # self.logger.info(self.config.mineru_api_token)
        # self.logger.info(self.config.mineru_base_url)
        # self.logger.info(mineru_config.base_url_prod)
        # self.logger.info(mineru_config.api_token_prod)

        #1.参数校验
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)
        #2.将pdf上传到mineru并轮询结果（batch_id 会登记到 runtime/mineru_pending.json，超时可捞回）
        zip_url = self._step_2_upload_and_poll(pdf_path_obj, state)
        #3.下载zip包并解压
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)
        #4.将.md文件的内容读取出来
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        #5.返回结果
        state["md_content"] = md_content
        state["md_path"] = md_path

        return state

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        """
       步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
       核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
       参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
       返回：最终MD文件的字符串格式绝对路径
       异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
       """
        response = requests.get(zip_url)
        if response.status_code != 200:
            raise FileProcessingError("下载ZIP包失败")
        #定义包的保存路径
        zip_save_path = output_dir_obj / f"{pdf_stem}.zip"
        with open(zip_save_path, "wb") as f:
            f.write(response.content)
        self.log_step(
            step_name="下载ZIP包",
            message=f"ZIP包下载成功，保存路径：{zip_save_path}"
        )

        #解压ZIP包
        extract_target_dir = output_dir_obj / pdf_stem

        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)

        extract_target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        self.log_step(
            step_name="ZIP下载",
            message=f"ZIP包解压成功，目标目录：{extract_target_dir}"
        )
        target_md_file = extract_target_dir / "full.md"
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)
        return str(new_md_path.absolute())







    def _step_2_upload_and_poll(self, pdf_path_obj: Path, state: ImportGraphState):
        """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：配置校验 → 获取上传链接 → 文件上传 → 登记batch_id → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象；state-流程状态（取task_id/file_dir用于登记）
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ConfigurationError(配置缺失)、PdfConversionError(请求/上传失败)、TimeoutError(轮询超时)
        设计要点：轮询超时不等于解析失败。MinerU 侧任务仍在排队，因此超时只把登记表标成
        timeout_pending 并抛出异常，事后可以用 scripts/mineru_probe.py --pending --resume 捞回结果。
        """
        # 1、配置文件校验
        if not mineru_config.mineru_base_url:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_BASE_URL 参数")
        if not mineru_config.mineru_api_token:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 参数")

        # 2、从MinerU服务器获取上传链接
        #组织上传需要的数据
        token = mineru_config.mineru_api_token
        url = f"{mineru_config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            # 公共云 vlm 通道容量紧张易排队，可用 MINERU_MODEL_VERSION=pipeline 切通道
            "model_version": mineru_config.model_version
        }
        file_path = ["demo.pdf"]

        #发送请求
        response = requests.post(url, headers=header, json=data)
        if response.status_code != 200:
            raise PdfConversionError(message=f"获取上传链接响应失败：状态码：{response.status_code}，响应结果：{response}")

        #获取响应结果
        result = response.json()
        if result["code"] !=0:
            raise PdfConversionError(message="获取上传链接失败. 返回数据：{}".format(result["msg"]))

        # 获取批量上传的batch_id和文件url
        batch_id = result["data"]["batch_id"]
        urls = result["data"]["file_urls"]

        # 先登记再上传：即使后续 PUT 失败或进程崩溃，batch_id 也不会丢
        record_batch(
            batch_id=batch_id,
            file_name=pdf_path_obj.name,
            task_id=state.get("task_id") or "",
            file_dir=state.get("file_dir") or "",
            import_file_path=str(pdf_path_obj),
            state=STATE_PENDING,
        )

        with open(pdf_path_obj, "rb") as f:
            res_upload = requests.put(urls[0], data=f)
            if res_upload.status_code != 200:
                raise PdfConversionError(message=f"{urls[0]}文件上传失败")

            self.logger.info(f"文件上传成功")

        #获取解析结果
        poll_url = f"{mineru_config.mineru_base_url}/extract-results/batch/{batch_id}"

        #轮询获取（超时与间隔改为配置项：MinerU 公共云高峰期排队常超 10 分钟）
        start_time = time.time()
        timeout_seconds = mineru_config.poll_timeout_seconds  # 最大超时时间
        poll_interval = mineru_config.poll_interval_seconds  # 轮询间隔时间
        self.log_step(
            step_name="轮询开始",
            message=f"[轮询开始]，轮询间隔:{poll_interval}秒，最大超时时间:{timeout_seconds}秒,batch_id:{batch_id}")
        while True:
            elapsed_time = time.time() -start_time
            if elapsed_time > timeout_seconds:
                # 我方放弃等待，但 MinerU 侧仍在排队/解析：标 timeout_pending 供事后捞回
                update_state(batch_id, STATE_TIMEOUT, err_msg=f"轮询超时 {int(elapsed_time)}s")
                raise TimeoutError(
                    f"[任务轮询]超时，已消耗时间：{int(elapsed_time)}s,batch_id:{batch_id}"
                    f"｜任务仍在 MinerU 队列中，无需重新上传，稍后执行："
                    f"uv run python -m scripts.mineru_probe --pending --resume"
                )

            #获取任务结果
            try:
                poll_res = requests.get(poll_url, headers=header, timeout=10)
            except Exception as exc:
                # 这里必须用 except 捕获到的异常对象（原代码误 from cmath import e，日志只会打印 2.718...）
                self.logger.warning(f"网络请求异常, {poll_interval}s后重试, batch_id:{batch_id}, 异常信息：{exc}")
                time.sleep(poll_interval)
                continue

            if poll_res.status_code != 200:
                raise PdfConversionError(f"[任务轮询]失败，状态码：{poll_res.status_code}")


            poll_res_json = poll_res.json()
            if poll_res_json["code"] != 0:
                raise PdfConversionError(f"[任务轮询]失败，错误码：{poll_res_json['code']}")

            extract_results = poll_res_json["data"]["extract_result"]
            if not extract_results:
                self.logger.warning(f"MinerU 返回结果列表为空，稍后重试, batch_id:{batch_id}")
                time.sleep(poll_interval)
                continue

            # 一批可能含多个文件：按文件名匹配取自己那条，避免硬取 [0] 拿到别人的结果
            extract_result = next(
                (r for r in extract_results if r.get("file_name") == pdf_path_obj.name),
                extract_results[0],
            )

            #获取任务状态值
            data_state = extract_result["state"]
            if data_state == "done":
                update_state(batch_id, STATE_DONE)
                self.log_step(
                    step_name="任务轮询",
                    message=f"解析完成，总耗时：{int(elapsed_time)}s,batch_id:{batch_id}"
                )
                full_zip_url = extract_result["full_zip_url"]
                self.log_step(
                    step_name="任务轮询",
                    message=f"获取全量ZIP地址成功：{full_zip_url},总耗时：{int(elapsed_time)},batch_id:{batch_id}"
                )
                return full_zip_url
            elif data_state == "failed":
                err_msg = extract_result.get("err_msg", "")
                update_state(batch_id, STATE_FAILED, err_msg=err_msg)
                raise PdfConversionError(f"[任务轮询]失败，错误信息：{err_msg}")

            else:
                self.log_step(
                    step_name="任务轮询",
                    message=f"任务进行中...已耗时：{int(elapsed_time)},状态：{data_state},batch_id:{batch_id}"
                )
                time.sleep(poll_interval)


    def _step_1_validate_paths(self, state: ImportGraphState):

        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(
                field_name="import_file_path",
                message="PDF路径不能为空",
                expected_type=str
            )
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(
                field_name="file_dir",
                message="输出路径不能为空",
                expected_type=str
            )
        #转换为path对象
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        #pdf是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(
                message=f"文件{pdf_path_obj.name}不存在",
            )

        #输出目录不存在则创建
        if not file_dir_obj.exists():
            self.logger.info(f"输出目录不存在，自动创建：{file_dir_obj.absolute()}")
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_dir_obj



if __name__ == "__main__":

    setup_logging()

    init_state = {
        "pdf_path": r"D:\youxi\wangke\04-设备手册汇总\doc\hak180产品安全手册.pdf",
        "file_dir": r"D:\youxi\wangke\output"
    }
    node_pdf_to_md = NodePDFToMD()
    result = node_pdf_to_md(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))