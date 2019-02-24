kill -9 $(lsof -t -i:5000)
echo killed pre
python3 recognize_gesture_keras.py&
cd ..
cd gui
npm start

kill -9 $(lsof -t -i:5000)
echo killed
