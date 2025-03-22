from pymilvus import connections

# Connect to Milvus instance (default localhost and port)
connections.connect(host='localhost', port='19530')

print("Connected to Milvus successfully!")
