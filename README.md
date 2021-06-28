# PeopleBehindDetector
[![Pylint](https://github.com/NetherQuartz/PeopleBehindDetector/actions/workflows/pylint.yml/badge.svg)](https://github.com/NetherQuartz/PeopleBehindDetector/actions/workflows/pylint.yml)

Итоговый проект первого семестра Deep Learning School — веб-приложение для обнаружения людей, стоящих позади пользователя.

## :bookmark_tabs: План выполнения
Я выбрал первый сценарий работы, так что мой план выглядит следующим образом:

- [X] Выбор фреймворка/библиотеки для использования детектора 
- [X] Запуск детектора на случайных изображениях
- [X] Выбор фреймворка/библиотеки для разработки веб/мобильного демо
- [X] Разработка демо
- [X] Встраивание модели-детектора в демо
- [X] Тестирование демо
- [X] Оформления демо для показа другим людям

## Небольшой отчёт
Модель — `ssdlite320_mobilenet_v3_large` из `torchvision`, она мало весит и достаточно быстро работает. Само веб-приложение написано с использованием библиотеки `streamlit`.

В приложении есть возможность как протестировать модель на любых изображениях, загружая их с диска, так и запустить детекцию людей на видео с вебкамеры. Люди обводятся красными прямоугольниками с подписью степени уверенности модели. Пороговая уверенность задаётся слайдером.

## Демо
Демо доступно на [Heroku](https://people-behind-detector.herokuapp.com/) и [Streamlit Sharing](https://share.streamlit.io/netherquartz/peoplebehinddetector/main/main.py). Прошу иметь в виду, что из-за сильной задержки вебкамера может не работать или работать не так, как ожидается. В этом случае рекомендую запустить приложение локально.

## Локальный запуск
1. Установите Python 3.8
2. Склонируйте проект: `git clone https://github.com/NetherQuartz/PeopleBehindDetector.git`
### Вариант 1. Poetry
```Bash
$ pip3 install poetry
$ poetry install
$ ./run.sh
```

### Вариант 2. requirements.txt
```Bash
$ pip install -r requirements.txt
$ streamlit run main.py
```
![demo](demo.gif)
