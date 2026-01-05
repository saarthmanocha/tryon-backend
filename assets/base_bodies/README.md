# Base Body Templates Directory

This directory contains photorealistic base body templates used for virtual try-on.

## Directory Structure

```
base_bodies/
├── male/
│   ├── slim.png
│   ├── regular.png
│   ├── chubby.png
│   └── muscular.png
├── female/
│   ├── slim.png
│   ├── regular.png
│   ├── chubby.png
│   └── muscular.png
└── neutral/
    ├── slim.png
    ├── regular.png
    ├── chubby.png
    └── muscular.png
```

## Template Requirements

Each template image should:

1. **Resolution**: 768x1024 pixels (portrait orientation)
2. **Subject**: Full body, front-facing, neutral pose
3. **Background**: Plain white or light gray (#F5F5F5)
4. **Clothing**: Minimal, tight-fitting neutral clothing (tank top + shorts)
5. **Lighting**: Soft, even lighting from front
6. **Quality**: Photorealistic, high quality (not mannequin/CGI)

## Where to Get Templates

Option 1: **Stock Photos**

- Getty Images, Shutterstock, Adobe Stock
- Search: "full body portrait neutral pose white background"
- Ensure license allows AI/commercial use

Option 2: **AI Generated** (if photorealistic)

- Midjourney, DALL-E, Stable Diffusion
- Prompt: "photorealistic full body portrait, [gender], [body type], neutral pose, white background, minimal clothing, soft lighting"

Option 3: **Custom Photography**

- Hire models for each body type
- Use consistent lighting setup
- Most reliable for quality

## Important Notes

- Templates should match IDM-VTON's training distribution (realistic photos)
- Avoid stylized, cartoon, or obvious CGI images
- Remove any logos/brand marks from clothing
- Face will be replaced, but body proportions matter
