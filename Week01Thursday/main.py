import time
import pyautogui
pyautogui.PAUSE = .1

# for el in range(1000):
#     time.sleep(.01)
#     print(pyautogui.position())


pyautogui.click(1273, 423)

def all():
    pyautogui.keyDown('ctrl')
    pyautogui.press('right')
    pyautogui.keyUp('ctrl')

    time.sleep(1)

    pyautogui.keyDown('command')
    pyautogui.press('t')
    pyautogui.keyUp('command')

    pyautogui.typewrite('www.zillow.com', .01)
    pyautogui.press('enter')

    time.sleep(3)

    pyautogui.typewrite('Charleston sc', .01)
    pyautogui.press('enter')

    time.sleep(2)

    pyautogui.click(942, 364)

    time.sleep(2)

    # button7location = pyautogui.locateOnScreen('a.png')
    # print(button7location)
    # print(pyautogui.center(button7location))
    #
    # button7x, button7y = pyautogui.center(button7location)
    #
    # pyautogui.click(button7x, button7y)

    for_sale = pyautogui.locateOnScreen('for_sale.png', grayscale=True)
    print(for_sale)
all()