@ECHO OFF
setlocal
call C:\Users\pinojc\Miniconda3\condabin\conda.bat activate pysb38
set PYTHONPATH=C:\Users\pinojc\PycharmProjects\PycharmProjects\pysb;%PYTHONPATH%
jupyter notebook
endlocal
