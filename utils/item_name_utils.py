# utils/item_name_utils.py
"""
商品名归一化工具
====================================================================
要解决的真实问题（上次导入体检挖出来的）：
    同一个产品（HAK180 烫金机）在库里留下了三个名字：
        BrotherHAK180烫金机 / 兄弟HAK180烫金机 / SCHNHAK180烫金机(孤儿)
    根因是「每个文件独立调用大模型识别商品名，且 prompt 对命名格式没有任何约束」，
    于是同一产品在不同手册里被识别成不同品牌写法（中英文混用、拼音缩写幻觉）。
    名字一分裂，检索按商品名聚类的召回被稀释，切片还会长成没人认领的孤儿。

本模块提供：
    1. normalize_item_name：结构化清洗（全角转半角、剥离文档类型词/公司名/噪声后缀、品牌别名统一）
    2. extract_model：抽型号锚点（会先剥掉疑似拼音品牌前缀）
    3. same_product / pick_existing_name：判断两个名字是否指同一个产品，并复用库里已有的写法

设计原则：宁可不合并，也不要错误合并。
    错误的归并会把两个产品的切片打到同一个商品名下，比名字分裂更难排查；
    所以 same_product 要求「型号相同」且「中文品类有 ≥2 字公共子串」两个条件同时成立。
    拿不准的情况（疑似拼音缩写品牌）只打 warning、保留原值，不做猜测性改名。
"""

import re
from typing import Dict, List, Optional

# 品牌别名表：key 统一小写，value 是入库使用的正式品牌名（只收确定等价的中英文对照）
BRAND_ALIASES: Dict[str, str] = {
    "brother": "兄弟", "兄弟": "兄弟",
    "hp": "惠普", "hewlettpackard": "惠普",
    "huawei": "华为", "h3c": "新华三", "newh3c": "新华三",
    "schneider": "施耐德", "siemens": "西门子", "omron": "欧姆龙",
    "keyence": "基恩士", "panasonic": "松下", "mitsubishi": "三菱",
    "sanyo": "三洋", "canon": "佳能", "epson": "爱普生",
}

# 文档类型词：属于文件名噪声，商品名里不该有（「品牌+型号+品类」才是商品名）
DOC_TYPE_WORDS: List[str] = [
    "产品安全手册", "使用说明书", "使用说明", "用户手册", "安装手册", "操作手册",
    "维护手册", "安全手册", "快速指南", "用户指南", "产品指南", "服务指南",
    "说明书", "手册", "指南", "的使用", "使用", "用户", "文档", "资料",
    "manual", "guide", "datasheet", "brochure",
]

# 公司/机构后缀：模型常把版权行里的公司名当成商品名的一部分
COMPANY_WORDS: List[str] = [
    "股份有限公司", "有限公司", "科技股份有限公司", "集团股份公司", "工业集团",
    "公司", "集团", "股份",
]

# 型号中的数字主体（同型号必有数字段，这是唯一可靠的锚点）
_DIGIT_RUN = re.compile(r"\d{2,6}")
# 数字段左侧最多回追几个字母当型号前缀
# 为什么是 3：正则「左起+贪婪」会把粘连的品牌一起吃进来（实测 SCHNHAK180 会被抽成
# NHAK180 而非 HAK180），固定窗口取「紧贴数字的最后 3 个字母」能把品牌噪声剪掉：
#   SCHNHAK180 → HAK180（与兄弟HAK180烫金机同一型号）
#   LaserJetProMFP1188 → MFP1188（而非 PROMFP1188）
# 代价：型号前缀超过 3 个字母时会丢开头（BMXCPT3231 → CPT3231），但两侧算法一致，
# 同一个产品不管怎么写都归到同一个值上，作为比较键只需要稳定不需要好看
_MODEL_LEFT_WINDOW = 3
# 中文连续段（品类语义就在这里）
_CN_SEGMENT = re.compile(r"[\u4e00-\u9fa5]+")


def _to_half_width(text: str) -> str:
    """全角字符转半角（含全角空格），避免 ＨＰ / HP 被当成两个名字"""
    chars = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        chars.append(chr(code))
    return "".join(chars)


def normalize_item_name(raw: Optional[str]) -> str:
    """
    结构化清洗商品名（只去确定是噪声的部分，不改语义）
    顺序：空值处理 → 全角转半角/去空白/去装饰符 → 剥公司后缀 → 剥文档类型词 → 品牌别名统一
    :param raw: 大模型返回的原始商品名
    :return: 清洗后的商品名；清洗后为空返回空串（由调用方决定兜底）
    """
    if not raw or not isinstance(raw, str):
        return ""

    name = _to_half_width(raw)
    name = re.sub(r"\s+", "", name)
    name = name.strip("\"'“”‘’*《》()（）").strip()
    if not name:
        return ""

    # 1. 公司后缀：只剥尾部，避免误伤正文
    changed = True
    while changed:
        changed = False
        for word in COMPANY_WORDS:
            if name.endswith(word) and len(name) > len(word):
                name = name[: -len(word)]
                changed = True
        if name.endswith("科技") and len(name) > 4:
            name = name[:-2]
            changed = True

    # 2. 文档类型词：长词优先，尾部整体剥离，中部出现则剔除（"的使用"、"手册"这类）
    for word in sorted(DOC_TYPE_WORDS, key=len, reverse=True):
        if name.lower().endswith(word.lower()) and len(name) > len(word):
            name = name[: -len(word)]
        elif word.isascii() and word.lower() in name.lower() and len(name) > len(word) + 2:
            name = re.sub(re.escape(word), "", name, flags=re.IGNORECASE)

    # 3. 品牌别名统一成中文正式名（中英混写是上次「同品三名」的主因）
    for alias, standard in BRAND_ALIASES.items():
        if alias.isascii() and re.search(re.escape(alias), name, re.IGNORECASE):
            name = re.sub(re.escape(alias), standard, name, flags=re.IGNORECASE)
        elif not alias.isascii() and alias in name:
            name = name.replace(alias, standard)

    name = re.sub(r"[\s_\-—·、,，。.]+$", "", name)
    return name.strip()


def _leading_upper_run(name: str) -> str:
    """取名字开头的连续大写段（判断疑似拼音品牌前缀用）"""
    match = re.match(r"^[A-Z]+", name or "")
    return match.group(0) if match else ""


def _find_model(text: str) -> str:
    """
    以数字段为锚抽型号：紧贴数字左侧最多 3 个字母 + 数字段，大写、去连字符
    多个数字段时取数字最长的那个（长度更能代表型号），同长取靠前的
    示例：HAK180→HAK180、SCHNHAK180→HAK180、RS-12→RS12、LA2608→LA2608、5000W→5000
    :return: 型号串；找不到数字段返回空串
    """
    if not text:
        return ""
    best = ""
    best_digits_len = 0
    for match in _DIGIT_RUN.finditer(text):
        digits = match.group()
        # 向左回追字母窗口，遇到连字符跳过，遇到中文/其他字符就停
        letters = ""
        idx = match.start() - 1
        while idx >= 0 and len(letters) < _MODEL_LEFT_WINDOW:
            ch = text[idx]
            if ch in "-\u2014· ":
                idx -= 1
                continue
            if ch.isascii() and ch.isalpha():
                letters = ch + letters
                idx -= 1
                continue
            break
        candidate = f"{letters}{digits}"
        if len(digits) > best_digits_len or (len(digits) == best_digits_len and not best):
            best, best_digits_len = candidate, len(digits)
    return best.upper()


def extract_model(name: str) -> str:
    """
    抽取型号（同一产品的强锚点）
    先清洗再抽型，保证品牌别名/文档噪声不影响型号结果
    """
    return _find_model(normalize_item_name(name))


def suspect_pinyin_brand(name: str) -> bool:
    """
    疑似「拼音缩写/幻觉品牌 + 型号粘连」：开头大写段长度 >4 且不在品牌别名表
    （典型如 SCHNHAK180，SCHN 不是任何已知品牌）。只用于打 warning 提示人工复核，
    绝不据此猜测性改名 —— 猜错品牌比留下脏名代价更高。
    """
    cleaned = normalize_item_name(name)
    run = _leading_upper_run(cleaned)
    if len(run) <= 4:
        return False
    return run.lower() not in BRAND_ALIASES


def _cn_tokens(name: str) -> List[str]:
    """取名字里的中文连续段（品类语义），并去掉已知品牌词，剩下的大体就是品类"""
    cleaned = normalize_item_name(name)
    for standard in set(BRAND_ALIASES.values()):
        cleaned = cleaned.replace(standard, "")
    return [seg for seg in _CN_SEGMENT.findall(cleaned) if seg]


def _longest_common_sublen(a: str, b: str) -> int:
    """两个字符串的最长公共子串长度（名字都很短，直接DP即可）"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def same_product(name_a: str, name_b: str) -> bool:
    """
    判断两个商品名是否指同一个产品（归一化的判定核心）
    规则（两条同时成立才算同一产品，防止误合并）：
        1. 型号相同（大小写/连字符差异已归一）
        2. 中文品类存在 ≥2 字的公共子串（"万用表的使用" 与 "数字万用表" 共享 "万用表"）
    双方都抽不到型号时，退化为「清洗后全名相等」
    """
    if not name_a or not name_b:
        return False
    clean_a, clean_b = normalize_item_name(name_a), normalize_item_name(name_b)
    if not clean_a or not clean_b:
        return False
    if clean_a == clean_b:
        return True

    model_a, model_b = extract_model(name_a), extract_model(name_b)
    if not model_a or not model_b or model_a != model_b:
        return False

    tokens_a, tokens_b = _cn_tokens(name_a), _cn_tokens(name_b)
    if not tokens_a or not tokens_b:
        # 品类信息缺失时不敢合并（型号相同但一个有品类一个没品类，交由人工判断）
        return False
    return max(_longest_common_sublen(x, y) for x in tokens_a for y in tokens_b) >= 2


def pick_existing_name(item_name: str, existing_names: List[str]) -> Optional[str]:
    """
    在已有商品名里找「同一个产品」的那个，用于复用库里写法，避免同品多名
    :param item_name: 本次识别（已清洗）的商品名
    :param existing_names: kb_item_names 里已有的商品名列表
    :return: 命中返回已存在的名字；未命中返回 None
    """
    if not item_name or not existing_names:
        return None
    for existing in existing_names:
        existing = str(existing or "").strip()
        if existing and same_product(item_name, existing):
            return existing
    return None


def canonical_key(name: str) -> str:
    """
    仅供日志/调试展示的稳定键（真正的匹配逻辑在 same_product）
    形如 "HAK180|烫金机"
    """
    cleaned = normalize_item_name(name)
    if not cleaned:
        return ""
    model = extract_model(name)
    tokens = _cn_tokens(name)
    return f"{model}|{''.join(tokens)}"
