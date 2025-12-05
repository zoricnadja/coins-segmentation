import os   
import sys
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

def process_folder(folder_path):
    image_paths = []
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            image_paths.append(file_path)
        elif filename.endswith('.csv'):
            csv_path = os.path.join(folder_path, filename)
            with open(csv_path, 'r') as csv_file:
                for line in csv_file:
                    image_name, coin_value = line.strip().split(',')
                    image_path = os.path.join(folder_path, image_name)
                    results.append((image_path, coin_value))
    return image_paths, results

def find_total_score(image_paths):
    total = {}
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is not None:
            total[image_path] = 0
          
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            mask_yellow = cv2.inRange(image_hsv, (19,113,60), (37,255,255))
            filtered_yellow_mask = cv2.medianBlur(mask_yellow, 5)   
            # plt.imshow(filtered_yellow_mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
            opening = cv2.morphologyEx( filtered_yellow_mask, cv2.MORPH_CLOSE, kernel, iterations=5)

            # plt.imshow(opening)
            # plt.show()
            yellow_coins = []
            star_coins = []
            contours, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
            # plt.imshow(image)
            # plt.show()
            """
            izdvajanje kontura uradjeno uz pomoc asistenta uz dodatne modifikacije
            """
            for c in contours:
                if len(c) < 5:
                    # print("Contour has less than 5 points, skipping.")
                    continue
                ellipse = cv2.fitEllipse(c)
                (x, y), (MA, ma), angle = ellipse
                area = cv2.contourArea(c)
                if area < 1250:
                    # print(f"Small area, skipping.{area}, {x}, {y}, {MA}, {ma}")
                    continue

                perimeter = cv2.arcLength(c, True)  
                if perimeter == 0:
                    continue

                if MA == 0 or ma == 0:
                    continue

                axis_ratio = min(MA, ma) / max(MA, ma) 

                ellipse_area = np.pi * (MA/2) * (ma/2)
                area_ratio = area / ellipse_area

                circularity = 4 * np.pi * area / (perimeter * perimeter)

                if (axis_ratio > 0.8 and 0.9 < area_ratio and circularity > 0.77) or (0.65 < axis_ratio <= 0.8 and 0.8 < area_ratio and circularity > 0.72 ) or (0.5 < axis_ratio <= 0.65 and 0.6 < area_ratio and 0.4 >circularity > 0.3 and area < 4000 ):
                    if area < 5000:
                        yellow_coins.append(c)
                        # print(f"Axis ratio: {axis_ratio}, Area ratio: {area_ratio}, Circularity: {circularity} izabrano {area}")

                    else:
                        star_coins.append(c)
                        # print(f"Axis ratio: {axis_ratio}, Area ratio: {area_ratio}, Circularity: {circularity} izabrano {area}")
                # else:
                    # print(f"Axis ratio: {axis_ratio}, Area ratio: {area_ratio}, Circularity: {circularity} odbijeno {area}, {x}, {y}, {MA}, {ma}")
            # cv2.drawContours(image, yellow_coins, -1, (255, 0, 0), 10)
            # cv2.drawContours(image, star_coins, -1, (0, 255, 255), 10)

            mask1 = cv2.inRange(image_hsv, (0, 120, 50), (10, 255, 255))
            mask2 = cv2.inRange(image_hsv, (170, 120, 50), (179, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
            filtered_red_mask = cv2.medianBlur(mask, 5) 
            diameter = 15
            kernel = np.ones((diameter, diameter), np.uint8)
            opening = cv2.morphologyEx(filtered_red_mask, cv2.MORPH_OPEN, kernel)

            # plt.imshow(opening)
            contours, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            red_coins = []
            """
            izdvajanje kontura uradjeno uz pomoc asistenta uz dodatne modifikacije
            """
            for c in contours:
                if len(c) < 5:
                    # print("Contour has less than 5 points, skipping.")
                    continue

                area = cv2.contourArea(c)
                if area < 700:
                    continue

                perimeter = cv2.arcLength(c, True)  
                if perimeter == 0:
                    continue

                ellipse = cv2.fitEllipse(c)
                (x, y), (MA, ma), angle = ellipse

                if MA == 0 or ma == 0:
                    continue

                axis_ratio = min(MA, ma) / max(MA, ma)

                ellipse_area = np.pi * (MA/2) * (ma/2)
                area_ratio = area / ellipse_area

                circularity = 4 * np.pi * area / (perimeter * perimeter)

                if (axis_ratio > 0.8 and 0.9 < area_ratio and circularity > 0.85) or (0.6 < axis_ratio <= 0.8 and 0.8 < area_ratio and circularity > 0.8):
                    red_coins.append(c)
                    # print(f"Axis ratio: {axis_ratio}, Area ratio: {area_ratio}, Circularity: {circularity}")

            # cv2.drawContours(image, red_coins, -1, (255, 0, 255), 10)
            # print(f"Detected {len(red_coins)} red coins.")
            # print(f"Detected {len(yellow_coins)} yellow coins.")
            # print(f"Detected {len(star_coins)} star coins.")
            # plt.imshow(image)
            # plt.show()
            total[image_path] += (len(red_coins) * 2 + len(yellow_coins) + len(star_coins) * 5)
        else:
            # print(f"Failed to load image: {image_path}")
            continue
    return total

"""
generisano pomocu asistenta
"""
def calculate_mae(predictions, ground_truth):
    errors = []
    for img_name, pred_value in predictions.items():
        true_value = ground_truth.get(img_name, 0)
        errors.append(abs(pred_value - int(true_value)))
    return np.mean(errors)

if __name__ == "__main__":
    folder_path = sys.argv[1]
    image_paths, results = process_folder(folder_path)    
    total = find_total_score(image_paths)
    mae = calculate_mae(total, dict(results))
    print(mae)