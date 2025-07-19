package com.example.tensorflowdemo

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.tensorflowdemo.databinding.ActivityMainBinding
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.nio.ByteBuffer
import java.nio.ByteOrder


class MainActivity : AppCompatActivity() {
    lateinit var interpreter:Interpreter;
    var inputImageWidth:Int= 0;
    var inputImageHeight:Int=0;
    var modelInputSize: Int = 0 // will be inferred from TF Lite model.
    private val FLOAT_TYPE_SIZE = 4
    private val PIXEL_SIZE = 1
    lateinit var binding:ActivityMainBinding;
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
         binding= ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val context:Context=this
        val model= FileUtil.loadMappedFile(context, "tflitemodel.tflite")
        val interpreter= Interpreter(model)
        val inputShape = interpreter.getInputTensor(0).shape()
        this.inputImageWidth = inputShape[1]
        this.inputImageHeight = inputShape[2]
        Log.i("width",(this.inputImageWidth).toString())
        Log.i("height",this.inputImageWidth.toString())
        val dtype = interpreter.getInputTensor(0).dataType()
        Log.i("inputDtype", dtype.toString())
        modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth *
                inputImageHeight * PIXEL_SIZE
        this.interpreter=interpreter;

        var btn=binding.button
        btn.setOnClickListener{
            val intent= Intent(MediaStore.ACTION_PICK_IMAGES)
            startActivityForResult(intent,1)
        }


    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode === RESULT_OK) {
            // compare the resultCode with the
            // constant
            if (requestCode === 1) {
                // Get the url of the image from data
                val selectedImageUri: Uri = data?.data!!
                if (null != selectedImageUri) {
                    // update the image view in the layout
                    //binding.imageView.setImageURI(selectedImageUri)
                    val bitmap=MediaStore.Images.Media.getBitmap(this.getContentResolver(),selectedImageUri)
                    binding.textView.text=classify(bitmap)

                }
            }
        }
    }
    override fun onDestroy(){
        super.onDestroy()
        this.interpreter.close()
    }

    private fun classify(bitmap: Bitmap): String {

        // TODO: Add code to run inference with TF Lite.
        // Pre-processing: resize the input image to match the model input shape.
        val imageProcessor =
            ImageProcessor.Builder()
                .add(ResizeOp(inputImageHeight, inputImageWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f,255f))
                .build()
        //binding.imageView.setImageBitmap(bitmap)
        val myTensorImage= TensorImage(DataType.FLOAT32)
        val rgbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        myTensorImage.load(rgbBitmap)
        val myimg=imageProcessor.process(myTensorImage)
        // Define an array to store the model output.
        val output = Array(1) { FloatArray(3) }
        Log.i("width",myimg.width.toString())
        Log.i("height",myimg.height.toString())
        Log.i("colorSpace",myimg.getColorSpaceType().toString())

        interpreter?.run(myimg.getTensorBuffer().buffer, output)

        // Post-processing: find the digit that has the highest probability
        // and return it a human-readable string.
        val result = output[0]
        Log.i("output", output[0].toString())
        Log.i("output", output[0][0].toString())
        Log.i("output", output[0][1].toString())
        Log.i("output", output[0][2].toString())
        val maxIndex = result.indices.maxByOrNull { result[it] } ?: -1
        val classes=arrayOf("Broccoli","Cauliflower","Unknown")
        val resultString =
            "Prediction Result: %s\nConfidence: %2f"
                .format(classes[maxIndex], result[maxIndex])

        return resultString
    }


}