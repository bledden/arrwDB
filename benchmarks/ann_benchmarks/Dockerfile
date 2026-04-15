# Dockerfile for ann-benchmarks arrwDB integration
FROM ann-benchmarks

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin
RUN pip install maturin numpy

# Copy arrwDB source
COPY . /home/app/arrwdb/
WORKDIR /home/app/arrwdb

# Build Rust HNSW
RUN cd rust/indexes && maturin build --release && \
    pip install target/wheels/*.whl

# Copy adapter
RUN cp benchmarks/ann_benchmarks/arrwdb_adapter.py \
    /home/app/ann_benchmarks/algorithms/arrwdb/module.py && \
    touch /home/app/ann_benchmarks/algorithms/arrwdb/__init__.py
