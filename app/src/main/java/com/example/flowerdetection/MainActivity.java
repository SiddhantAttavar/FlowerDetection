package com.example.flowerdetection;

//AndroidX Import
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

//Android imports
import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;


//Graphics imports. Eg. Bitmap
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

//Importing TensorFlow Lite
import org.tensorflow.lite.Interpreter;

//Importing modules for reading files
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.channels.FileChannel;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.AbstractMap;
import java.util.Map;

//Some data structures. Eg. Lists etc
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.PriorityQueue;
import java.util.Comparator;

//Date and Time modules
import java.text.SimpleDateFormat;
import java.util.Date;

public class MainActivity extends AppCompatActivity {

    Bitmap myBitmap;//Initializing variable to store the bitmap image captured by the camera

    Interpreter tflite;// Declaring Tensorflow Lite Interpreter
    protected ByteBuffer imgData = null;//Declaring ByteBuffer to store the image

    int DIM_IMG_SIZE_X = 150;//Initializing the width of the image in pixels to be 150
    int DIM_IMG_SIZE_Y = 150;//Initializing the height of the image in pixels to be 150
    int DIM_PIXEL_SIZE = 3;//Initializing the no. of color channels: 3, i.e. RGB

    private int[] intValues;

    private List<String> labelList;//List of class names

    private final Interpreter.Options tfliteOptions = new Interpreter.Options();//Interpreter options

    private float[][] labelProbArray = null;//Array for TF output

    // array that holds the labels with the highest probabilities
    private String[] topLables = null;
    // array that holds the highest probabilities
    private String[] topConfidence = null;

    private static final int RESULTS_TO_SHOW = 3;

    // activity elements
    private ImageView selected_image;
    private Button capture_button;
    private TextView label1;

    File photoFile = null;//The captured image file
    static final int CAPTURE_IMAGE_REQUEST = 1;

    String mCurrentPhotoPath;
    private static final String IMAGE_DIRECTORY_NAME = "FLOWER_DETECTION";//The directory to save the image

    // priority queue that will hold the top results from the CNN
    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            //Call functions to load the model and the labels from their files
            tflite = new Interpreter(loadModelFile(), tfliteOptions);
            labelList = loadLabelList();
        } catch (Exception e) {
            e.printStackTrace();
        }

        //Allocates storage for imgData
        imgData = ByteBuffer.allocateDirect(4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());

        //Initializing size of labelProbArray
        labelProbArray= new float[1][labelList.size()];

        //Sets size of intValues
        intValues = new int[4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

        // labels that hold top results of CNN
        label1 = (TextView) findViewById(R.id.label1);

        // displays the probabilities of top labels
        // initialize imageView that displays selected image to the user and Button
        selected_image = (ImageView) findViewById(R.id.selected_image);
        capture_button = (Button) findViewById(R.id.capture_button);

        topLables = new String[3];
        topConfidence = new String[3];

        //Defines button on click behaviour
        capture_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                    captureImage();
                }
                else {
                    captureImage2();
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //Once image is captured, Bitmap is converted to ByteBuffer and inference is done
        if(data.getData()==null){
            myBitmap = (Bitmap)data.getExtras().get("data");
            selected_image.setImageBitmap(myBitmap);
            ByteBuffer image = convertBitmapToByteBuffer(myBitmap);
            doInference(image);
            selected_image.setImageBitmap(myBitmap);
        }else{
            try {
                myBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        if (requestCode == CAPTURE_IMAGE_REQUEST && resultCode == RESULT_OK && data.getData()!=null) {
            Bitmap myBitmap = BitmapFactory.decodeFile(photoFile.getAbsolutePath());
            ByteBuffer image = convertBitmapToByteBuffer(myBitmap);
            doInference(image);
            selected_image.setImageBitmap(myBitmap);
        }
    }

    private void captureImage() {
        //Checks permissions, captures the image from camera (Android version >= Lollipop) and calls function to save the image in a file
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[] { Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE }, 0);
        }
        else
        {
            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                // Create the File where the photo should go
                try {

                    photoFile = createImageFile();
                    displayMessage(getBaseContext(),photoFile.getAbsolutePath());
                    Log.i("Mayank",photoFile.getAbsolutePath());

                    // Continue only if the File was successfully created
                    if (photoFile != null) {
                        Uri photoURI = FileProvider.getUriForFile(this,
                                "com.example.flowerdetection.fileprovider",
                                photoFile);
                        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                        startActivityForResult(takePictureIntent, CAPTURE_IMAGE_REQUEST);
                    }
                } catch (Exception ex) {
                    // Error occurred while creating the File
                    displayMessage(getBaseContext(),ex.getMessage().toString());
                }


            }else
            {
                displayMessage(getBaseContext(),"Nullll");
            }
        }
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        mCurrentPhotoPath = image.getAbsolutePath();
        return image;
    }



    private void captureImage2() {
        //Captures the image from camera (Android version < Lollipop) and calls function to save the image in a file
        try {
            Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
            photoFile = createImageFile4();
            if(photoFile!=null)
            {
                displayMessage(getBaseContext(),photoFile.getAbsolutePath());
                Uri photoURI  = Uri.fromFile(photoFile);
                cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(cameraIntent, CAPTURE_IMAGE_REQUEST);
            }
        }
        catch (Exception e)
        {
            displayMessage(getBaseContext(),"Camera is not available."+e.toString());
        }
    }

    private File createImageFile4() {
        //Creates file to store the image
        // External sdcard location
        File mediaStorageDir = new File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                IMAGE_DIRECTORY_NAME);
        // Create the storage directory if it does not exist
        if (!mediaStorageDir.exists()) {
            if (!mediaStorageDir.mkdirs()) {
                displayMessage(getBaseContext(),"Unable to create directory.");
                return null;
            }
        }

        //Saves image in format yearmonthdate_hourminutessecondsIMG_time.jpg
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss",
                Locale.getDefault()).format(new Date());
        File mediaFile = new File(mediaStorageDir.getPath() + File.separator
                + "IMG_" + timeStamp + ".jpg");

        return mediaFile;

    }

    // converts bitmap to byte array which is passed in the tflite graph
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        //Converts the captured image from a Bitmap to a ByteBuffer so that predictions can be made
        if (imgData == null) {
            return null;
        }
        imgData.rewind();

        //converts the image
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // loop through all pixels and save result in imgData
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                // get rgb values from intValues where each int holds the rgb values for a pixel.
                // if quantized, convert each rgb value to a byte, otherwise to a float

                imgData.put((byte) ((val >> 16) & 0xFF));
                imgData.put((byte) ((val >> 8) & 0xFF));
                imgData.put((byte) (val & 0xFF));

            }
        }

        return imgData;
    }

    public void doInference (ByteBuffer imgData) {
        //Calls the tf.lite.Interpreter.run() function to make predictions and calls a function to display the results
        tflite.run(imgData, labelProbArray);
        printTopKLabels();
    }

    // print the top labels and respective confidences
    private void printTopKLabels() {
        //Displays the most likely result
        // add all results to priority queue
        for (int i = 0; i < labelList.size(); ++i) {
            sortedLabels.add(new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }

        // get top results from priority queue
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            topLables[i] = label.getKey();
            topConfidence[i] = String.format("%.0f%%", label.getValue() * 100);
        }

        label1.setText("You are really confusing me. I think it is "+topLables[2]);
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        //Loads the model.tflite file from the assets folder and returns it as a MappedByteBuffer
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // loads the labels from the label txt file in assets into a string array
    private List<String> loadLabelList() throws IOException {
        //Loads the labels.txt file, reads it and returns a list of the class names
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(this.getAssets().open("labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private void displayMessage(Context context, String message) {
        //Function to display text as a toast
        Toast.makeText(context,message,Toast.LENGTH_LONG).show();
    }

}
