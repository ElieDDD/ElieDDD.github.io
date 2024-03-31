// set global - needed for external libraries
/* globals ml5 */

const message = document.querySelector("#message")
const fileButton = document.querySelector("#file")
const img = document.querySelector("#img")
const synth = window.speechSynthesis

fileButton.addEventListener("change", event => loadFile(event))
img.addEventListener("load", () => userImageUploaded())

function loadFile(event) {
  img.src = URL.createObjectURL(event.target.files[0])
}

function userImageUploaded() {
  message.innerHTML = "Image was loaded!"
}

function speak(text) {
  if (synth.speaking) {
    return
  }
  if (text !== "") {
    let utterThis = new SpeechSynthesisUtterance(text)
    synth.speak(utterThis)
  }
}