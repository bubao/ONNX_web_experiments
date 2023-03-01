var js_img= {};
window.js_img = js_img;
var results;
let imgElement = document.getElementById("imageSrc")
let inputElement = document.getElementById("fileInput");
let session;
let pyodide = {}
//https://github.com/Kazuhito00/AnimeGANv2-ONNX-Sample/blob/main/AnimeGANv2_Convert2ONNX.ipynb
async function animegan2() {
  // Pyodide is now ready to use...
  console.log(pyodide.runPython(`
    import sys
    sys.version
  `));
  var test=""
  await pyodide.loadPackage(["numpy", "pillow"])
  await pyodide.runPython(`
    from js import window
    from PIL import Image
    import numpy as np
    import base64
    import io
    import re
    side_length=512

    msg = re.sub('^data:image/.+;base64,', '', window.js_img)
    msg = base64.b64decode(msg)

    buf = io.BytesIO(msg)
    img=Image.open(buf).convert("RGB")
    width, height = img.size

    img=img.resize((side_length,side_length),Image.BICUBIC)
    # print(img)
    x = np.array(img, dtype=np.float32)
    x = x.transpose(2, 0, 1)
    x = x * 2 - 1
    x = x.reshape(-1, 3, side_length, side_length)

    x=x.flatten()
    # print(x)
    `)
  var array=pyodide.globals.get('x').toJs();
  var width=pyodide.globals.get('width');
  var height=pyodide.globals.get('height');
  console.log(width, height);
  // const session = await ort.InferenceSession.create('http://127.0.0.1:8080/face_paint_512_v2_0.onnx');
  //var input_name = session.get_inputs()[0].name
  //output_name = session.get_outputs()[0].name
  console.log(array)
  const tensor = new ort.Tensor('float32', array, [1,3,512,512]);
  console.log(tensor)
  const feeds = { "input_image": tensor};
  results = await session.run(feeds);
  results= results["output_image"].data
  console.log(results)
  pyodide.globals.set("results", results);
  pyodide.globals.set("height", height);
  pyodide.globals.set("width", width);
  await pyodide.runPython(`
       #from js import results,width,height
       import numpy as np
       from PIL import Image
       import base64
       onnx_result = np.array(list(results))
       onnx_result =onnx_result.reshape(1,3,512,512)

       onnx_result = onnx_result.squeeze()
       # print(onnx_result)
       onnx_result = (onnx_result * 0.5 + 0.5).clip(0, 1)
       onnx_result = onnx_result * 255
       onnx_result = onnx_result.transpose(1, 2, 0).astype('uint8')

       onnx_result=onnx_result.flatten()

       onnx_result =onnx_result.reshape((512,512,3))
       onnx_result = Image.fromarray(np.uint8(onnx_result)).resize((width,height))

       #print("xsdw",onnx_result)

       buffered = io.BytesIO()
       onnx_result.save(buffered, format="PNG")
       img_byte = buffered.getvalue()

       # 将字节流编码成base64字符串
       onnx_result = base64.b64encode(img_byte).decode('utf-8')
    `)
  var onnx_res = pyodide.globals.get('onnx_result')
  pyodide.globals.delete("results")
  console.log("onnx_res",onnx_res)
  // var len = onnx_res.byteLength;
  // let binary=""
  // for (var i = 0; i < len; i++) {
  //   binary += String.fromCharCode( onnx_res[ i ] );
  // }
  var image="data:image/png;base64,"+onnx_res;
  console.log(image)
  document.getElementById("output").src = image
};

window.onload = async function() {
  session = await ort.InferenceSession.create('/face_paint_512_v2_0.onnx');
  pyodide = await loadPyodide({ indexURL : "https://cdn.jsdelivr.net/pyodide/v0.19.0/full/" });

  inputElement.addEventListener("change", (e) => {
    imgElement.src = URL.createObjectURL(e.target.files[0]);
    var reader = new FileReader();
    reader.readAsDataURL(e.target.files[0]);
    reader.onload = function () {
      window.js_img=reader.result
      // console.log(reader.result);
      animegan2()
    }
  }, false);
}



//https://github.com/onnx/models/blob/master/vision/super_resolution/sub_pixel_cnn_2016/dependencies/Run_Super_Resolution_Model.ipynb
async function Upscale() {
  let pyodide = await loadPyodide({ indexURL : "https://cdn.jsdelivr.net/pyodide/v0.19.0/full/" });
  // Pyodide is now ready to use...
  console.log(pyodide.runPython(`
    import sys
    sys.version
  `));

  await pyodide.loadPackage(["numpy", "pillow"])
  await pyodide.runPython(`
    from js import js_img
    from PIL import Image
    import numpy as np
    import base64
    import io
    import re
    side_length=224

    msg = re.sub('^data:image/.+;base64,', '', js_img)
    msg = base64.b64decode(msg)

    buf = io.BytesIO(msg)
    orig_img=Image.open(buf)
    width, height = img.size
    img = orig_img.resize((side_length,side_length),Image.BICUBIC)
    img_ycbcr = img.convert('YCbCr')
    img_y_0, img_cb, img_cr = img_ycbcr.split()
    img_ndarray = np.asarray(img_y_0)
    img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
    img_5 = img_4.astype(np.float32) / 255.0
    x = img_5.flatten()
    #print(x)
    `)
  var array=pyodide.globals.get('x').toJs();
  var width=pyodide.globals.get('width').toJs();
  var height=pyodide.globals.get('height').toJs();
  const session = await ort.InferenceSession.create('../super-resolution-10.onnx');
  console.log(array)
  const tensor = new ort.Tensor('float32', array, [1,3,512,512]);
  console.log(tensor)
  const feeds = { "input": tensor};
  results = await session.run(feeds);
  results= results["output"].data
  console.log(results)
  await pyodide.runPython(`
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
      "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
  `)
  var array=pyodide.globals.get('final_img').toJs();
  document.getElementById("output").src = array
};


