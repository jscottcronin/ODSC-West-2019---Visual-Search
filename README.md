# Lessons learned deploying a deep learning visual search service at scale
Scott Cronin, Ph.D.  
Senior Data Scientist  
ShopRunner

This repo goes alongside an ODSC West Talk from 2019.  
[ODSC West 2019 -- Presentation](./ODSC_West_2019_Presentation.pdf)

# Development
## Setup

The folowing code allows us to extract a feature vector from a local image file called `dress.jpeg`

```bash
# Clone the repo
git clone https://github.com/jscottcronin/odsc_west_2019_visual_search.git
cd odsc_west_2019_visual_search/
```

## Lesson 1 and 1.5. Use Docker
Locally:
```bash
docker build -t visual_search .
docker run -it --rm visual_search bash
```
On Cloud GPU Compute:
```bash
docker build -t visual_search .
nvidia-docker run -it --rm visual_search bash
```
Start jupyter from within container:
```bash
jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```
**Note: if on a cloud compute, you'll need the port 8888 opened on the instance 
