@echo off
rem Compile verification for final/ + older/ against FTC SDK 11.2.1 + Pedro 2.1.2.
rem Uses the local JDK and Maven-extracted jars previously staged in ftc_compile.
rem Usage: compile_check.cmd              (compile only)
rem         compile_check.cmd -runmath    (also run the BallMath Java self-test)
rem         compile_check.cmd -runwrapper (also run the wrapper logic self-test)

setlocal enabledelayedexpansion
set ROOT=%~dp0
set COMPILE_DIR=%LOCALAPPDATA%\ftc_compile
set JDK=%COMPILE_DIR%\jdk\jdk-17.0.20.1+1\bin
set LIB=%COMPILE_DIR%\mvn\lib
if not exist "%JDK%\javac.exe" (echo JDK not found under %COMPILE_DIR% & exit /b 1)
if not exist "%LIB%\ftc_robotcore.jar" (echo jars not found under %LIB% & exit /b 1)

set CP=%LIB%\ftc_hardware.jar;%LIB%\ftc_robotcore.jar;%LIB%\pedro_core.jar;%LIB%\pedro_ftc.jar
set OUT=%COMPILE_DIR%\out_final
if not exist "%OUT%" mkdir "%OUT%"

for /r "%ROOT%final" %%f in (*.java) do set SRC=!SRC! "%%f"
for /r "%ROOT%older" %%f in (*.java) do set SRC=!SRC! "%%f"
for /r "%ROOT%wrapper" %%f in (*.java) do set SRC=!SRC! "%%f"

"%JDK%\javac.exe" -cp "%CP%" -d "%OUT%" %SRC%
if errorlevel 1 (echo COMPILE FAILED & exit /b 1)
echo COMPILE OK - classes in %OUT%

if /i "%~1"=="-runmath" (
    "%JDK%\java.exe" -cp "%OUT%" org.firstinspires.ftc.teamcode.BallMath
)
if /i "%~1"=="-runwrapper" (
    "%JDK%\java.exe" -cp "%OUT%;%CP%" org.firstinspires.ftc.teamcode.wrapper.WrapperLogicTest
)
endlocal