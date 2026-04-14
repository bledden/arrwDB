"""
gRPC server for arrwDB.

Provides lower-latency access than REST for high-throughput search workloads.
Uses protobuf binary serialization and HTTP/2 multiplexing.

To generate Python stubs from the proto file:
    python -m grpc_tools.protoc -I app/grpc --python_out=app/grpc --grpc_python_out=app/grpc app/grpc/arrwdb.proto

To start the gRPC server alongside the REST API:
    python -m app.grpc.server --port 50051
"""
