import pymongo
from datetime import datetime

class SRLogger:
    def __init__(self):
       self.client = pymongo.MongoClient("mongodb://localhost:27017/")
       self.db = self.client["SmartRouterDB"]
       self.collection = self.db["logs"] # Collection for storing logs
        
    def write_log(self, session_id,  event_name, data):
        entry = {
            "session_id": session_id,
            "event_name": event_name,
            "data": data,
            "timestamp": datetime.now()
        }
        self.collection.insert_one(entry)
 
    
    def retrieve_logs(self, session_id):
        query = {"session_id": session_id}
        return list(self.collection.find(query).sort("timestamp", pymongo.ASCENDING))
    
    def retrieve_last_20_logs(self):
        # Retrieve the last 10 logs
        return list(self.collection.find().sort("timestamp", pymongo.DESCENDING).limit(10))