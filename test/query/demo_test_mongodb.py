from pymongo import MongoClient

myclient = MongoClient("mongodb://192.168.2.8:27017/")
mydb = myclient["atguigudb"]

dblist = myclient.list_database_names()
if "runoobdb" in dblist:
    print("数据库已存在")

print(dblist)

mydb = myclient["runoobdb"]
mycollection = mydb["sites"]

mylist = [
  {"name": "Taobao", "alexa": "100", "url": "https://www.taobao.com"},
  {"name": "QQ", "alexa": "101", "url": "https://www.qq.com"},
  {"name": "Facebook", "alexa": "10", "url": "https://www.facebook.com"},
  {"name": "知乎", "alexa": "103", "url": "https://www.zhihu.com"},
  {"name": "Github", "alexa": "100", "url": "https://www.github.com"}
]

x = mycollection.insert_many(mylist)

print(x)