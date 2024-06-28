const inputImage = document.getElementById('inputImage');
const uploadButton = document.getElementById('processImage');
const outputImage = document.getElementById('outputImage');
const inputImageDisplay = document.getElementById('inputImageDisplay');

uploadButton.addEventListener('click', () => {
    const imagePath = inputImage.files[0].path;
    inputImageDisplay.src = imagePath;
    window.electronAPI.processImage(imagePath);
});

window.electronAPI.getImage((event, data) => {
    // outputImage.src = data;
    // console.log('data:image/jpeg;base64,' + data);
    // outputImage.src = 'data:image/jpeg;base64,' + data;
    // Convert the base64 data to binary and create a blob URL
    // Convert the base64 data to binary and create a blob URL
    const binaryString = atob(data);
    const byteArray = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        byteArray[i] = binaryString.charCodeAt(i);
    }
    const blob = new Blob([byteArray], { type: 'image/jpeg' });
    console.log(blob);
    outputImage.src = URL.createObjectURL(blob);
});