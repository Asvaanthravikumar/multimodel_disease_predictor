@echo off
title MesoPredict AI - Multimodal Early Disease Prediction System
echo ======================================================================
echo  MESOPREDICT AI - MULTIMODAL EARLY DISEASE PREDICTION SYSTEM
echo  Clinical Decision Support & Explainable AI Engine
echo ======================================================================
echo.
echo Starting Flask AI Application Server...
echo Launching browser at http://127.0.0.1:5000 in 3 seconds...
start "" http://127.0.0.1:5000
python app.py
pause
