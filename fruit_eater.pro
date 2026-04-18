QT -= gui

CONFIG += c++14 console # c++14 is important for tensorflow!
CONFIG -= app_bundle

SOURCES += \
        main.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += $$PWD/../../../../../../usr/include/SDL2         # SDL2
INCLUDEPATH += $$PWD/../../../../../../usr/include/tensorflow/  # tensorflow

INCLUDEPATH += $$PWD/../../../../../../usr/include/tensorflow/include/src/ # for protobuf
INCLUDEPATH += $$PWD/../../../../../../usr/include/tensorflow/include/ # ?

LIBS += -L/usr/lib -lSDL2 -lSDL2_ttf

unix:!macx: LIBS += -L$$PWD/../../../../../usr/local/lib/ -ltensorflow_cc

HEADERS += \
    defines.h \
    utils.h


