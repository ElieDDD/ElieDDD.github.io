let facemesh;
let video;
let predictions = [];
var r;
var g;
var b;
var a;
let font,
  fontsize = 32;

function preload() {
  // Ensure the .ttf or .otf font stored in the assets directory
  // is loaded before setup() and draw() are called
  font = loadFont('norwester.otf');
}


function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(width/2, height/2);
   textFont(font);
  textSize(fontsize);
  //textAlign(CENTER, CENTER);
     
  r = random(255); // r is a random number between 0 - 255
  g = random(255); // g is a random number betwen 100 - 200
  b = random(255); // b is a random number between 0 - 100
  a = random(200,255); // a is a random number between 200 - 255

  facemesh = ml5.facemesh(video, modelReady);
img = loadImage('LB.jpg'); 
  // This sets up an event that fills the global variable "predictions"
  // with an array every time new predictions are made
  facemesh.on("predict", results => {
    predictions = results;
  });

  // Hide the video element, and just show the canvas
  video.hide();
}

function modelReady() {
  console.log("Model ready!");
}

function draw() {

  image(video, 0, 0, width/2, height/2);

  // We can call both functions to draw all keypoints
  drawKeypoints();
   fill(0);
  text('What does it take to not be a face?', 5, 300);
   text('Try costumes, masks, disguises', 5, 325);
  image(img, 0, 330, img.width, img.height);
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  for (let i = 0; i < predictions.length; i += 1) {
    const keypoints = predictions[i].scaledMesh;

    // Draw facial keypoints.
    for (let j = 0; j < keypoints.length; j += 1) {
      const [x, y] = keypoints[j];
fill(r, g, b);
      //fill(0, 255, 0);
      ellipse(x/2, y/2, 5, 5); //perfect
    }
  }
}
