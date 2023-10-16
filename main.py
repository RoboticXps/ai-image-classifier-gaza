from teachable_machine import TeachableMachine
import cv2 as cv
import serial

ser = serial.Serial("/dev/ttyACM0", 115200)
model_path = "model/keras_model.h5"
labels_path = "model/labels.txt"
image_path = "screenshot.jpg"

cap = cv.VideoCapture(0)

model = TeachableMachine(model_path=model_path,
                         labels_file_path=labels_path)

B_state = False
F_state = False
classify_per_frame_counter = 0

def select_microbit_emotions(class_name):
    if "Bomb" in class_name and B_state:
        ser.write(b'55')
        print("Bomb state sent to micro:bit")
    if "Flag" in class_name and F_state:
        ser.write(b'155')
        print("Flag state sent to micro:bit")


while True:
    _, img = cap.read()
    cv.imwrite(image_path, img)

    result = model.classify_image(image_path)

    # print("class_index", result["class_index"])

    print("class_name:::", result["class_name"])

    # print("class_confidence:", result["class_confidence"])

    cv.imshow("Video Stream", img)

    if classify_per_frame_counter >= 10:
        classify_per_frame_counter = 0
        if "B" in result["class_name"][0]:
            B_state = True
            F_state = False
        if "F" in result["class_name"][0]:
            F_state = True
            B_state = False
        select_microbit_emotions(result["class_name"])
    else:
        # sleep(0.1)
        classify_per_frame_counter+=1

    k = cv.waitKey(1)
    print(classify_per_frame_counter)

    # If Esc is pressed -> close the program
    if k % 255 == 27:
        break

# ser.close()

