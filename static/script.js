const dropArea = document.querySelector('.drag-area');
const dragText = document.querySelector('.header');

let button = dropArea.querySelector('.button');
let input = dropArea.querySelector('input');

let file;

button.onclick = () => {
  input.click();
};

function loading_on() {
  document.querySelector('.overlay').style.display = "block";
}

function loading_off() {
  document.querySelector('.overlay').style.display = "none";
}

// when browse
input.addEventListener('change', function () {
  file = this.files[0];
  dropArea.classList.add('active');
  displayFile();
});

// when file is inside drag area
dropArea.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropArea.classList.add('active');
  dragText.textContent = 'Release to Upload';
  // console.log('File is inside the drag area');
});

// when file leave the drag area
dropArea.addEventListener('dragleave', () => {
  dropArea.classList.remove('active');
  // console.log('File left the drag area');
  dragText.textContent = 'Drag & Drop';
});

// when file is dropped
dropArea.addEventListener('drop', (event) => {
  event.preventDefault();
  // console.log('File is dropped in drag area');

  file = event.dataTransfer.files[0]; // grab single file even of user selects multiple files
  // console.log(file);
  displayFile();
  // window.alert(file && file['type'].split('/')[0] === 'image');
});

document.querySelector('form').addEventListener('submit', function(e) {
  if (file == null){
    alert('Please upload your image!');
  }
  else{
  e.preventDefault();
  const formData = new FormData();
  formData.append('file', file);
  const xhr = new XMLHttpRequest();

  xhr.onreadystatechange = function() {
    if (xhr.readyState == XMLHttpRequest.DONE && xhr.status == 200) {
      // window.alert(xhr.status)
      loading_off()
      const data = JSON.parse(xhr.responseText).data;
      const update =  new Date();
      document.querySelector('.person__img').innerHTML = `<img src="/static/src/0.jpg?v=${update.getTime()}" />`; // To update avoid using image from cache
      document.querySelector('.info__id').innerHTML = `ID: ${data[1]}`;
      document.querySelector('.info__name').innerHTML = `Full name: ${data[2]}`;
      document.querySelector('.info__date').innerHTML = `Date of birth: ${data[3]}`;
      document.querySelector('.info__sex').innerHTML = `Sex: ${data[4]}`;
      document.querySelector('.info__nation').innerHTML = `Nationality: ${data[5]}`;
      document.querySelector('.info__hometown').innerHTML = `Place of origin: ${data[6]}`;
      document.querySelector('.info__address').innerHTML = `Place of residence: ${data[7]}`;
      document.querySelector('.info__doe').innerHTML = `Date of expiry: ${data[8]}`;
    }
    else if (xhr.status == 404){
      const data = JSON.parse(xhr.responseText).data;
      window.alert(data)
      loading_off()
    }
  }
  const URL = '/uploader';
  xhr.open('POST', URL);
  xhr.send(formData);
}
});


function displayFile() {
  let fileType = file.type;
  // console.log(fileType);

  let validExtensions = ['image/jpeg', 'image/jpg', 'image/png'];

  if (validExtensions.includes(fileType)) {
    // console.log('This is an image file');
    let fileReader = new FileReader();

    fileReader.onload = () => {
      let fileURL = fileReader.result;
      // console.log(fileURL);
      let imgTag = `<img src="${fileURL}" alt="">`;
      dropArea.innerHTML = imgTag;
    };
    fileReader.readAsDataURL(file);
  } else {
    alert('This file is not supported!');
    dropArea.classList.remove('active');
  }
}