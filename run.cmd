@echo off
rem Pure-Java autograd demos. No TensorFlow DLL required.
rem Usage:
rem   run.cmd                        -> XOR + regression demos (puretf.demo.Demo)
rem   run.cmd fashion [epochs=5]     -> Fashion-MNIST MLP classifier
rem   run.cmd fashion-cnn [epochs=5] -> Fashion-MNIST CNN (add --fast for a subset)
rem   run.cmd gradcheck              -> finite-difference gradient verification
rem   run.cmd features               -> integration smoke test (all layer stacks)
setlocal
set "MAVEN_OPTS=-Xmx1g"
cd /d "%~dp0"

if /i "%~1"=="fashion" goto fashion
if /i "%~1"=="fashion-cnn" goto fashioncnn
if /i "%~1"=="gradcheck" goto gradcheck
if /i "%~1"=="features" goto features

mvn -q compile exec:java %*
goto :eof

:fashion
shift
set "EPOCHS=%~1"
if "%EPOCHS%"=="" set EPOCHS=5
mvn -q compile "-Dexec.mainClass=puretf.demo.FashionMnistDemo" exec:java "-Dexec.args=C:\Users\dell\Desktop\data\fashion %EPOCHS%"
goto :eof

:fashioncnn
shift
set "REST=%*"
if "%REST%"=="" (set "ARGS=C:\Users\dell\Desktop\data\fashion 5") else (set "ARGS=C:\Users\dell\Desktop\data\fashion %REST%")
mvn -q compile "-Dexec.mainClass=puretf.demo.CnnFashionMnistDemo" exec:java "-Dexec.args=%ARGS%"
goto :eof

:gradcheck
mvn -q compile "-Dexec.mainClass=puretf.check.GradCheck" exec:java
goto :eof

:features
mvn -q compile "-Dexec.mainClass=puretf.check.FeaturesTest" exec:java
goto :eof
