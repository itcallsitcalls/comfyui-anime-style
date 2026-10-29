# comfyui-anime-style

给 ComfyUI 写的两个自定义节点，用来处理二次元风格的成片。AnimeStylePost 做赛璐璐风格的调色后处理，BatchCurate 给一批图打分排序、挑出最好的几张。

## 解决什么问题

AI 出图有两个常见毛病：一是色彩和光影不稳定，同一批图风格容易漂；二是一次生成几十张，废图要人工一张张看。

AnimeStylePost 把赛璐璐画风的几个关键点做成可调参数，用来统一成片风格。BatchCurate 用清晰度、色彩、曝光、构图四个指标给整批图排序，先过滤掉明显不行的，剩下的再人工细挑。它做的是初筛，不代替审美，最终选哪张还是人说了算。

## 安装

把整个文件夹复制到 ComfyUI 的 custom_nodes 目录，重启 ComfyUI。只依赖 numpy、Pillow、torch，都是 ComfyUI 自带的，不用额外安装。

装好后节点在 image/style（AnimeStylePost）和 image/batch（BatchCurate）分类下。

## 节点

### AnimeStylePost

赛璐璐风格后处理，参数：

- posterize_levels：色调分离层数，越低色块越平，越接近平涂，6 到 10 比较合适。
- saturation：饱和度。
- contrast：对比度。
- brightness：亮度。
- lineart_strength：线稿增强强度，0.5 到 0.8 效果明显。
- glow：柔光光晕，处理逆光或风景时可以加到 0.3 到 0.5。

### BatchCurate

给一批图打分排序，输出 top_n 张，另有一个 report 字符串输出，写的是完整排序结果。

四个打分维度各有权重可以调：

- w_sharpness：清晰度，拉普拉斯方差。
- w_color：色彩，饱和度加色相多样性。
- w_exposure：曝光，惩罚过暗、过曝和削波。
- w_composition：构图，三分法交点处的边缘能量占比。

分数是批内相对分，只在同一批图里有比较意义，绝对数值没有参考价值。

## 示例工作流

workflows 目录里有两个工作流，拖进 ComfyUI 画布就能用。

workflow_single.json 是单图的，Load Image 接 AnimeStylePost 再接 Save Image，只用核心节点和这个包，不依赖别的插件。

workflow_pipeline.json 是完整管线：批量读图、风格化、打分筛选、保存，另外接了 Show Text 显示评分报告。这个工作流用到两个插件，需要先装：

- lhyNodes，提供 Load Image Batch 和 Image Batch to Images。Load Image Batch 输出的 image_batch 是插件自定义类型，必须经过 Image Batch to Images 转成标准 IMAGE 才能接后面的节点。
- ComfyUI-Custom-Scripts，提供 Show Text 节点，用来显示评分报告。不装也能跑，只是看不到文字报告。

用 pipeline 工作流时注意：Image Batch to Images 的 width 和 height 要填单张原图的实际尺寸，不是预览里显示的那个拼接外框。lhyNodes 的预览会把一批图拼起来显示，那个外框尺寸不代表真实分辨率。图片尺寸不统一的话，按尺寸分文件夹分批处理，别混在一起跑。

## License

MIT
