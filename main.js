const { app, BrowserWindow, Menu, shell, ipcMain } = require("electron");
const path = require('node:path');
const fs = require("fs");
const { spawn } = require('child_process');

const menuItems = [
    {
        label: "Menu",
        submenu: [
            {
                label: "Info",
                async click() {
                    const win2 = new BrowserWindow({
                        height: 300,
                        width: 400,
                        show: false,
                        // backgroundColor: '#2e2c29',
                    });

                    win2.loadFile('info.html');
                    win2.once('ready-to-show', () => {
                        win2.show();
                    });
                },
            },
        ],
    },
    {
        label: "File",
        submenu: [
            {
                label: "Exit",
                click() {
                    app.quit()
                },
            },
        ],
    },
    {
        label: "Window",
        submenu: [
            {
                role: "Minimize",
            },
            {
                role: "close",
            },
        ],
    },
];

const menu = Menu.buildFromTemplate(menuItems);
Menu.setApplicationMenu(menu);

const createWindow = () => {
    const win = new BrowserWindow({
        height: 500,
        width: 800,
        webPreferences: {
            nodeIntegration: true,
            preload: path.join(__dirname, 'preload.js')
        }
    });

    // win.webContents.openDevTools();
    win.loadFile('index.html');
};

app.whenReady().then( () => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    })
});

app.on('window-all-closed', () => {
    if (process.platform !== "darwin") app.quit();
})

ipcMain.on('process-image', (event, imagePath) => {
    const pythonScript = path.join(__dirname, 'model', 'run.py');
    
    const pythonProcess = spawn('python', [pythonScript, imagePath]);

    // const pythonExecutable = path.join(__dirname, 'dist', 'run.exe'); 

    // const pythonProcess = spawn(pythonExecutable, [imagePath]);

    console.log(imagePath);
    let imageData = '';

    pythonProcess.stdout.on('data', (data) => {
        // console.log('output received:', data);
        receivedData = data.toString();
        imageData += receivedData; 
        if (receivedData === '') {
            console.log('Received empty data!');
        }
    });

    pythonProcess.stdout.on('end', () => {
        console.log('All data received');
        console.log('imageData:', imageData);
        event.reply('image-processed', imageData);
    });

});