from ultralytics import YOLO
import os
import numpy as np
import cv2
import cvzone
import math
from analiza_imaginilor import sort
import tkinter as tk
from tkinter import filedialog


while True:

    video_to_be_selected = None
    photo_to_be_selected = None


    def select_mask():
        """
        Această funcție este folosită pentru a încărca o mască/imagine din fisierul de lucru.

        Imaginea trebuie sa aibă format .jpg sau .jpeg sau .png
        """
        global photo_to_be_selected
        try:
            photo_to_be_selected = filedialog.askopenfilename(
                initialdir="photo_samples", 
                title="Masca",
                filetypes=(("Imagine/poză", "*.jpg *.jpeg *.png"), ("All Files", "*.*")))
            if photo_to_be_selected:
                check_extension = photo_to_be_selected.lower()
                if check_extension.endswith((".png", ".jpeg", ".jpg")):
                    photo_label.config(text=f"Masca încărcată: {os.path.basename(photo_to_be_selected)}", bg="white", fg="black")
                else:
                    photo_label.config(text=f"EROARE: Încărcați un format valid (.png, .jpeg, .jpg)", bg="yellow", fg="red")
            check_validity()
        except Exception as e:
            print(f"Eroare la selectarea măștii: {str(e)}")


    def select_video():
        """
        Această funcție este folosită pentru a încărca un videoclip din fisierul de lucru.

        Imaginea trebuie sa aibă format .mp4
        """
        global video_to_be_selected
        try:
            video_to_be_selected = filedialog.askopenfilename(
                initialdir="video_samples",
                title="Video", 
                filetypes=(("Videoclip", "*.mp4"), ("All Files", "*.*")))
            if video_to_be_selected:
                check_video_extension = video_to_be_selected.lower()
                if check_video_extension.endswith((".mp4")):
                    video_label.config(text=f"Videoclip încărcat: {os.path.basename(video_to_be_selected)}", bg="white", fg="black")
                else:
                    video_label.config(text=f"EROARE: Încărcați un format valid (.mp4)", bg="yellow", fg="red")
            check_validity()
        except Exception as e:
            print(f"Eroare la selectarea videoclipului: {str(e)}")


    def check_validity():
        """
        Această funcție verifică dacă atât videoclipul, cât și imaginea au un format valid.
        """
        global video_to_be_selected, photo_to_be_selected
        if (video_to_be_selected and video_to_be_selected.lower().endswith(".mp4")) and \
        (photo_to_be_selected and photo_to_be_selected.lower().endswith((".png", ".jpeg", ".jpg"))):
            continue_button.config(state=tk.NORMAL)
        else:
            continue_button.config(state=tk.DISABLED)


    def send_ctrl_c():
        """
        Această funcție închide aplicația
        """
        if os.name == 'nt':
            os.kill(os.getpid(), 0x40000002)


    def continue_process():   
        """
        Această funcție este folosită pentru închide interfața.
        """
        root.destroy()


    root = tk.Tk()
    root.title("Alegerea măștii și a videoclipului")
    root.geometry("400x300")
    root.config(bg="slate blue")

    photo_button = tk.Button(root, text="Selectați masca", command=select_mask, width=15, height=2, bg="bisque", fg="black")
    photo_button.pack(pady=10)

    photo_label = tk.Label(root, text="Masca încărcată:", bg="white")
    photo_label.pack(pady=5)

    video_button = tk.Button(root, text="Selectați videoclipul", command=select_video, width=15, height=2, bg="bisque", fg="black")
    video_button.pack(pady=10)

    video_label = tk.Label(root, text="Videoclip încărcat:", bg="white")
    video_label.pack(pady=5)

    continue_button = tk.Button(root, text="Continuare", command=continue_process, width=15, height=2, bg="bisque")
    continue_button.pack(pady=10)
    continue_button.config(state=tk.DISABLED)

    stop_button = tk.Button(root, text="Închide", command=send_ctrl_c, width=15, height=2, bg="bisque")
    stop_button.pack(pady=5)

    root.mainloop()

    if photo_to_be_selected != None and video_to_be_selected != None:

        my_photo = os.path.basename(photo_to_be_selected)
        my_video = os.path.basename(video_to_be_selected)

        mask = cv2.imread(f"photo_samples/{my_photo}")
        cap = cv2.VideoCapture(f"video_samples/{my_video}")
        model = YOLO('yolo_weights/yolov8n.pt')

    tracker = sort.Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    limits = [175, 450, 1300, 420]
    totalCount = []

    classNames = ["persoana","bicicleta","masina","motocicleta","avion","autobuz", "tren", "camion","boat","semafor",
                "extinctor","semn stop","parcare", "banca","pasare","pisica","caine","cal","oaie","vaca",
                "elefant", "urs", "zebra","girafa","ghiozdan","umbrela","sacosa","cravata", "valiza","disc",
                "skiuri","snowboard","minge","zmeu","bata baseball","manusa baseball","skateboard", "placa de surf","racheta tenis","sticla",
                "pahar", "ceasca", "furculita","cutit","lingura", "bol","banana","mar","sandwich","portocala",
                "broccoli","morcov", "hot dog","pizza","gogoasa","tort","scaun","canapea","planta","pat",
                "masa","toaleta", "tv","laptop", "mouse","telecomanda","tastatura","telefon","cuptor cu microunde","cuptor",
                "toaster","chiuveta","frigider","carte","ceas","vaza","foarfeca","animal de plus","uscator de par","periuta",
                ]
    try:
        while True:

            success, img = cap.read()
            if not success:
                break
            imgRegion = cv2.bitwise_and(img, mask)

            if imgRegion is None:
                print("A apărut o eroare în procesarea măștii")
                break
            results = model(imgRegion, stream=True)
            detections = np.empty((0,5))

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2-x1, y2-y1
                    cvzone.cornerRect(img, (x1,y1,w,h), l=10)
                    conf = math.ceil((box.conf[0]*100))/100
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]
                    if currentClass == "masina" \
                        or currentClass == "camion" \
                        or currentClass == "bicicleta" \
                        or currentClass == "autobuz" \
                        or currentClass == "motocicleta" \
                        and conf > 0.5:
                        cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0,x1), max(25, y1)), scale=2, thickness=1, offset=0)
                        cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=5)
                        currentArray = np.array([x1,y1,x2,y2,conf])
                        detections = np.vstack((detections, currentArray))

            resultTracker = tracker.update(detections)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 5)

            for result in resultTracker:
                x1,y1,x2,y2,id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(result)
                cx, cy = x1+w//2,y1+h//2 
                if limits[0] << cx << limits[2] and limits[1]-50 < cy < limits[3]+50:
                    if totalCount.count(id) == 0:
                        totalCount.append(id)
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,0), 5)
            
            cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"{len(totalCount)}")
                cv2.destroyAllWindows()
                break
                
            print(f"Numarul total de vehicule este: {len(totalCount)}")
            cv2.imshow(f"{my_video}", img)
            cv2.imshow(f"{my_photo}", imgRegion)
            cv2.waitKey(0)
            
    except Exception as e:
        print(f"Selectați imaginea și videoclipul care urmează a fi testate")