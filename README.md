# YMCA APP
*** The human-pose model used in this app can only run on linux machines using the [Intel NCS](https://software.intel.com/en-us/neural-compute-stick)***

In this example we'll be using a simple set of cases to determine if someone is doing a Y, M, C, or A in our image
then displaying the letter on the screen if they do creating a fun virtual experience.

## Requirements
- An [alwaysAI account](https://www.alwaysai.co/auth?register=true)
- [alwaysAI installed](https://dashboard.alwaysai.co/docs/getting_started/development_computer_setup.html) on a development machine
- [Docker installed](https://dashboard.alwaysai.co/docs/getting_started/edge_device_setup.html on the target deployment device (if different than the development machin)

## Running
First configure where the app will be deployed (locally or to a remote device) with: `aai app configure`

Then deploy the code with `aai app deploy`

And finally to start: `aai app start`

Open `localhost:5000` on any browswer to view the output streamer.

To end the app just click on the red stop button in the streamer or CTRL+C from the command line.

## About AlwaysAI
AlwaysAI is a platform aimed at democratizing AI on the edge by making it easier. We provide clis, apis, a model 
catalog and docker containers to help you get started building a CV app in minutes. Visit our website at [alwaysi.co](https://www.alwaysai.co) for more info.