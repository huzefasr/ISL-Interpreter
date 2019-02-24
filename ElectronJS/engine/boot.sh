kill -9 $(lsof -t -i:5000)
echo killed pre
cd ..
cd gui
npm start
cd ..
cd engine
python3 recognize_gesture_keras.py
kill -9 $(lsof -t -i:5000)
echo killed
