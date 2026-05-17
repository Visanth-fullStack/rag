@echo off
echo Setting up Python environment...
python -m venv venv
call venv\Scripts\activate
pip install -r backend/requirements.txt
echo Setup complete. Run 'run_backend.bat' to start the service.
pause
