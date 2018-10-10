import pyautogui as pag
import time

pag.hotkey('ctrl', 'right')
time.sleep(1)
pag.hotkey('command', 't')
time.sleep(2)
pag.typewrite('charleston.craigslist.org', interval=.03)
pag.press('enter')
time.sleep(2)
# button = pag.locateOnScreen('a.png')
button_cars_x, button_cars_y = 714, 681
# x, y = pag.locateCenterOnScreen('a2.png')
time.sleep(1)
# print(x, y)


pag.click(button_cars_x, button_cars_y)
# pag.click(x,y)
time.sleep(2)

button_search_x, button_search_y = 292, 148
pag.click(button_search_x, button_search_y)
time.sleep(1)

pag.typewrite('westfalia', interval=.03)
time.sleep(3)
pag.press('enter')

# for el in range(1000):
#     time.sleep(.01)
#     print(pag.position())

