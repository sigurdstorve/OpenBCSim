/*
Copyright (c) 2015, Sigurd Storve
All rights reserved.

Licensed under the BSD license.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <QApplication>
#include <QDesktopWidget>
#include <QSurfaceFormat>
#include <QDebug>
#include "MainWindow.hpp"
#include "utils.hpp"

int main(int argc, char** argv) try {
    QApplication app(argc, argv);
    app.setApplicationDisplayName("Interactive OpenBC sim. interface");
    QSurfaceFormat fmt;
    fmt.setDepthBufferSize(24);
    //if (QCoreApplication::arguments().contains(QStringLiteral("--multisample"))) {
        qDebug() << "Using multisample";
        fmt.setSamples(4);
    //}
    // Hard-coded to use the core profile.
    fmt.setVersion(3, 2);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(fmt);
    
    MainWindow mainWindow;
    mainWindow.resize(mainWindow.sizeHint());
    int desktopArea = QApplication::desktop()->width()*QApplication::desktop()->height();
    int widgetArea = mainWindow.width()*mainWindow.height();
    if (((float)widgetArea / (float)desktopArea) < 0.75f) {
        mainWindow.show();
    } else {
        mainWindow.showMaximized();
    }
    return app.exec();
    
} catch (std::exception& e) {
    std::cout << "Caught exception: " << e.what() << std::endl;
} catch (...) {
    std::cout << "Caught unknown exception.\n";
}
