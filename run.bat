@echo off
chcp 65001 >nul
title Garbage Classification Demo
cd /d "D:\TH Học Sâu\garbage-demo"

echo ============================================
echo   HE THONG PHAN LOAI RAC - DANG KHOI DONG
echo ============================================
echo.

call venv\Scripts\activate

echo [1/2] Mo giao dien web (index.html)...
start "" index.html

echo [2/2] Khoi dong FastAPI server tai http://127.0.0.1:8000
echo.
echo Nhan CTRL+C de dung server.
echo ============================================
echo.

uvicorn app:app --reload

pause
