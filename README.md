### Welcome to the **AI Image Multiverse !**

### What’s new:

-   Updated **OpenAI** with newer models
    
-   Added **Google Gemini**
    
-   Added **AI Horde**
    
-   Added **Presets**
    

**Note:** Only image models that have  API keys in your `.env` file will appear in the dropdown menu

----------
## REQUIREMENTS:

### OpenAI

-   Since DALL·E 2 and DALL·E 3 are being deprecated, newer models were added
    
-   You’ll need a **paid** OpenAI API key to use these models (Register here: [OpenAI](https://openai.com/api/))
    

### Google Gemini

-   You’ll need a Gemini API key  to use these models
    
-   The **free** tier still requires a **credit card**
(Register here: [Google Gemini](https://aistudio.google.com/app/api-keys))

-   The models in the current code are pay-as-you-go (just like the Open AI models)

-   Create a **project** and a **billing account**, then **link** billing to the project — otherwise the API will return a **429** error
    
### AI Horde

-   A **free** image API — the tradeoff is slower speeds and older image models (Register here: [AI Horde](https://aihorde.net/register))
    
-   For the fastest results, pick "first available" model. The results will vary depending on what model is assigned
    
-   Most models are Stable Diffusion–based
    
-   **AlbedoBase XL (SDXL)** typically looks better, but often has high load. The dropdown shows current availability & current load; if load is too high, the quality upgrade is disabled.  It can often take 15 minutes to create an image.  Open terminal veiw to see updates.
    
-   To use the AI Horde **random** function, you’ll also need a **free** [Groq / Llama API key](https://console.groq.com/settings/organization)


## Screenshot

![screenshot](https://github.com/doowylloh88/InkyPi-Image-Multiverse/blob/main/ai_image_multiverse/docs/images/main.png)


## Installation

### Install

Install the plugin using the InkyPi CLI, providing the plugin ID & GitHub repository URL:

```bash
inkypi install doowylloh88 https://github.com/doowylloh88/InkyPi-Image-Multiverse
```

## Development-status

- This plug-in was 100% created using vibe- coding & a lot of yelling at ChatGPT.  An actual coder should take over the project to maintain it
- Huge Caveat: I'm in developer mode ONLY. I don't have an e-ink screen yet.  They are sold out everywhere.  As I wait for stocks to resupply,  I'm creating plug-ins I want. I'm running off a Raspberry Pi 5.  Most of the image processing happens on the Image Model side, so it shouldn't slam a Pi Zero 2W any more than the existing image plug-in


### License

This project is licensed under the GNU public License.
