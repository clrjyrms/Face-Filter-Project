import face_recognition
import numpy as np
import cv2

#Filter
filter_1 = cv2.imread("eyes.png", -1)
filter_2 = cv2.imread("pink_smoke.png", -1)

#Video Stream
print("Starting SnappedChat")
vs = cv2.VideoCapture(0)
vs.set(cv2.CAP_PROP_FPS, 30)

#Function for overlaying the filter
def apply_filter(frame, filtr, pos=(0,0), scale = 1): 

	filtr = cv2.resize(filtr, (0,0), fx=scale, fy=scale)
	h, w, _ = filtr.shape
	rows, cols, _ = frame.shape
	y, x = pos[0], pos[1]

	for i in range(h):
		for j in range(w):
			if x + i >= rows or y + j >= cols:
				continue

			alpha = float(filtr[i][j][3] / 255.0)
			frame[x + i][y + j] = alpha * filtr[i][j][:3] + (1 - alpha) * frame[x + i][y + j]
	return frame

#Loop video stream
while True:

	ret, frame = vs.read()

	image_frame = frame[:, :, ::-1]
	f_detected = face_recognition.face_locations(image_frame)
	face = [(0,0,0,0)]

	if f_detected != []:

		face = [[f_detected[0][3], f_detected[0][0], abs(f_detected[0][3] - f_detected[0][1]) + 100, abs(f_detected[0][0] - f_detected[0][2])]]

		for (x, y, w, h) in face:

			x -= 20
			w -= 60
			y += 40	
			
			filter_ymin = int(y - 1.5 * h / 5)
			filter_ymax = int(y + 2.5 * h / 5)
			sy_filter = filter_ymax - filter_ymin

			y -= 40

			filter_xmin = int(y + 4 * h / 6)
			filter_xmax = int(y + 5.5 * h / 6)
			sx_filter = filter_xmax - filter_xmin

			eye_frame = frame[filter_ymin:filter_ymax, x:x+w]

			x += 130

			lip_frame = frame[filter_xmin:filter_xmax, x:x+w]

			filtery_resized = cv2.resize(filter_1, (w, sy_filter),interpolation=cv2.INTER_CUBIC)
			filterx_resized = cv2.resize(filter_2, (w, sx_filter),interpolation=cv2.INTER_CUBIC)

			apply_filter(eye_frame, filtery_resized)
			apply_filter(lip_frame, filterx_resized)

	cv2.imshow("SnappedChat", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
print("Closing SnappedChat")
vs.release()
cv2.destroyAllWindows()
