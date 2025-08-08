build:
    @uv run maturin develop

build-release:
    @uv run maturin build --release
