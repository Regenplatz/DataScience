import numpy as np
import cv2 # install OpenCV with conda
from random import randint
import time

supermarket = cv2.imread('supermarket.png')

drinks = [150,200]
dairy = [520,200]
spices = [900,200]
fruits = [1300,200]
entrance = [1300,650]
checkout = [100,750]

class Customer:

    def __init__(self, positions):
        self.current_position = 0
        self.x = positions[self.current_position][0]
        self.y = positions[self.current_position][1]
        self.counter = 0

        self.next_x = positions[self.current_position+1][0]
        self.next_y = positions[self.current_position+1][1]
        self.positions = positions

        img_pos = randint(0, 4)
        images = ["mann1.png", "mann2.png", "mann3.png", "frau1.png", "frau2.png"]
        self.customer = cv2.imread(images[img_pos])  # once

    def set_target(self,pos):
        self.tx = pos[0]
        self.ty = pos[1]

    def move_target(self):
        vx = 0
        vy = 0
        distance_x = self.next_x - self.x
        distance_y = self.next_y - self.y

        if abs(distance_x) > abs(distance_y):
            distance = abs(distance_x)
        else:
            distance = abs(distance_y)


        if distance_x != 0:
            vx = round(distance_x / distance) * round((distance+50) / 50)  # go faster if farther away
        if distance_y != 0:
            vy = round(distance_y / distance) * round((distance+50) / 50)

        self.x = self.x + vx
        self.y = self.y + vy

        # check boundaries
        if self.x < 0:
            self.x = 0
        if self.x > 1460:
            self.x = 1460
        if self.y < 0:
            self.y = 0

        if distance < 2:
            next_pos = self.positions[self.current_position+1]

            if len(self.positions) > self.current_position+1 and self.positions[self.current_position+1] != checkout:
                self.current_position += 1
                self.x = self.positions[self.current_position][0]
                self.y = self.positions[self.current_position][1]
                self.counter = 0

                self.next_x = self.positions[self.current_position + 1][0]
                self.next_y = self.positions[self.current_position + 1][1]

    def draw(self, frame):
        """move and draw into frames"""
        frame[self.y:self.y+96, self.x:self.x+96] = self.customer
        return frame



def s_background():
    bg = np.zeros((890, 1521, 3), dtype=np.uint8)

    return bg

# load frame

def show_supermarket(customers):
    bg = s_background()

    # assign initial locations
    counter = 0

    # while True:
    while counter < 3000:
        counter += 1
        frame = bg.copy()
        frame[1:880,1:1521] = supermarket

        for customer_counter, s in enumerate(customers):
            if customer_counter * 50 < counter:
                s.move_target()
            frame = s.draw(frame)

        cv2.imshow('frame', frame)

        time.sleep(0.01)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def migrate_paths(cust_movements):
    customer_list = []
    for key in cust_movements:
        cust_path = []
        for station in cust_movements[key]:

            if station == 'drinks':
                cust_path.append(drinks)
            elif station == 'dairy':
                cust_path.append(dairy)
            elif station == 'spices':
                cust_path.append(spices)
            elif station == 'fruit':
                cust_path.append(fruits)
            elif station == 'entrance':
                cust_path.append(entrance)
            elif station == 'checkout':
                cust_path.append(checkout)

        customer_list.append(Customer(cust_path))

    return customer_list


if __name__ == "__main__":
    path1 = [entrance, dairy, fruits, spices, drinks, checkout]
    path2 = [entrance, spices, drinks, dairy, fruits, checkout]
    path3 = [entrance, drinks, checkout]

    customers = [
        Customer(path1),
        Customer(path2),
        Customer(path3),
    ]

    show_supermarket(customers)