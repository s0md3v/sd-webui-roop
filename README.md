# roop for StableDiffusion

This is an extension for StableDiffusion's [AUTOMATIC1111 web-ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) that allows face-replacement in images. It is based on [roop](https://github.com/s0md3v/roop) but will be developed seperately.

![example](example/example.png)

### Disclaimer

这个软件旨在为快速增长的人工智能生成媒体行业做出有效的贡献。它将协助艺术家完成一些任务，例如动画制作自定义角色或将该角色用作服装模特等。

该软件的开发者知道它可能存在的不道德应用，并致力于采取预防措施。它具有内置的检查机制，防止该程序用于不适当的媒体中。我们将继续朝着积极的方向开发这个项目，恪守法律和道德标准。如果根据法律要求，这个项目可能会被关闭或在输出上加上水印。

使用这个软件的用户应该在遵守当地法律的同时负责任地使用。如果使用了真实人物的面部，用户建议从有关人士获得同意，并在发布在线内容时清楚说明它是深度伪造的。该软件的开发者不对最终用户的行为负责。

## Installation
首先，如果由于某些原因无法安装，不要在此处开立问题事项。通过谷歌搜索你遇到的错误。

针对Windows，下载并安装Visual Studio。在安装过程中，确保安装了Python和C++程序包。

+ 运行此命令：`pip install insightface==0.7.3`
+ 在web-ui中，转到“扩展”选项卡，并在“从URL安装”选项卡中使用此URL `https://github.com/s0md3v/sd-webui-roop`
+ 关闭web-ui并重新运行它
+ 如果遇到" 'NoneType' object has no attribute 'get' "错误，请下载inwapper_128.onnx模型，并将其放置在`<webui_dir>/models/roop/`目录中。

对于其余的错误，请使用谷歌搜索。祝你好运。

## Usage

1. Under "roop" drop-down menu, import an image containing a face.
2. Turn on the "Enable" checkbox
3. That's it, now the generated result will have the face you selected

## Tips
#### Getting good quality results
First of all, make sure the "Restore Face" option is enabled. You can also try the "Upscaler" option or for more finer control, use an upscaler from the "Extras" tab.

For even better quality, use img2img with denoise set to `0.1` and gradually increase it until you get a balance of quality and resembelance.

#### Replacing specific faces
If there are multiple faces in an image, select the face numbers you wish to swap using the "Comma separated face number(s)" option.

#### The face didn't get swapped?
Did you click "Enable"?

If you did and your console doesn't show any errors, it means roop detected that your image is either NSFW or wasn't able to detect a face at all.
