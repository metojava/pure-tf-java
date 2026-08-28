@echo off
rem Pure-Java autograd demos. No TensorFlow DLL required.
rem Usage:
rem   run.cmd                        -> XOR + regression demos (puretf.demo.Demo)
rem   run.cmd fashion [epochs=5]     -> Fashion-MNIST MLP classifier
rem   run.cmd fashion-cnn [epochs=5] -> Fashion-MNIST CNN (add --fast for a subset)
rem   run.cmd gan                    -> Generative Adversarial Network on 2D data
rem   run.cmd lstm-text              -> character-level LSTM language model
rem   run.cmd gradcheck              -> finite-difference gradient verification
rem   run.cmd features               -> integration smoke test (all layer stacks)
rem   run.cmd formats                -> safetensors/ONNX/TF/Torch format interchange tests
rem   run.cmd safetensors            -> safetensors round-trip demo
rem   run.cmd onnx                   -> ONNX round-trip demo
setlocal
set "MAVEN_OPTS=-Xmx1g"
cd /d "%~dp0"

if /i "%~1"=="fashion" goto fashion
if /i "%~1"=="fashion-cnn" goto fashioncnn
if /i "%~1"=="gan" goto gan
if /i "%~1"=="lstm-text" goto lstmtext
if /i "%~1"=="gradcheck" goto gradcheck
if /i "%~1"=="features" goto features
if /i "%~1"=="formats" goto formats
if /i "%~1"=="safetensors" goto safetensors
if /i "%~1"=="onnx" goto onnx

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

:gan
mvn -q compile "-Dexec.mainClass=puretf.demo.GanDemo" exec:java
goto :eof

:lstmtext
mvn -q compile "-Dexec.mainClass=puretf.demo.LstmTextDemo" exec:java
goto :eof

:features
mvn -q compile "-Dexec.mainClass=puretf.check.FeaturesTest" exec:java
goto :eof

:formats
mvn -q compile "-Dexec.mainClass=puretf.check.FormatsTest" exec:java
goto :eof

:safetensors
mvn -q compile "-Dexec.mainClass=puretf.demo.SafeTensorsDemo" exec:java
goto :eof

:onnx
mvn -q compile "-Dexec.mainClass=puretf.demo.OnnxDemo" exec:java
goto :eof
