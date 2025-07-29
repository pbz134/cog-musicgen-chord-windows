@echo off
SET HF_HOME=%~dp0
SET TORCH_HOME=%~dp0
SET XFORMERS_DISABLE_TRITON=1
.\python-3.10.0-embed-amd64\python.exe app.py
pause
