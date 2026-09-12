---
name: Windows bundle compatibility
description: Compatibility rule for using the redesigned UI with the packaged Windows TensorRT runtime
---

The redesigned Gradio UI must adapt its video callback to the pipeline signature available in the local Windows bundle, passing values by supported parameter name instead of assuming positional parity.

**Why:** The packaged TensorRT runtime can contain an older pipeline that omits newer optional text-driving parameters. Passing the newer UI's full positional list shifts or overflows the callback even though TensorRT itself is healthy.

**How to apply:** Keep inference callbacks unchanged. Add new UI controls as optional trailing named inputs, inspect the bound pipeline signature at runtime, and omit parameters that the installed bundle does not support.