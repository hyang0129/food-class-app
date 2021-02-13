package com.foodclass.foodclassifier;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.TimeoutError;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import androidx.appcompat.app.AppCompatActivity;

import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;

import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.util.HashMap;


public class MainActivity extends AppCompatActivity {

    private ImageView imgCapture;
    private Button btnCapture;
    private TextView textView;
    private static final int Image_Capture_Code = 1;
    private Bitmap bp;
    private String url = "https://tpu-44747.uc.r.appspot.com/predict";
    private String urlwarmup = "https://tpu-44747.uc.r.appspot.com";
    private Integer warmupTries = 0;
//    private String url = "http://10.0.2.2:8080/predict";

    private RequestQueue mRequestQueue;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgCapture = (ImageView) findViewById(R.id.capturedImage);
        textView = (TextView) findViewById(R.id.textView);
        btnCapture =(Button)findViewById(R.id.btnTakePicture);
        btnCapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent cInt = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cInt, Image_Capture_Code);
                }
            }
        );
        mRequestQueue = Volley.newRequestQueue(getApplicationContext());

        textView.setText("Warming Up");

        warmUp();

        }

    public void warmUp(){
        StringRequest stringRequest = new StringRequest
                (Request.Method.GET, urlwarmup,  new Response.Listener<String>() {
                    @Override
                    public void onResponse(String response) {
                        textView.setText("Ready");
                    }
                }, new Response.ErrorListener() {

                    @Override
                    public void onErrorResponse(VolleyError error) {
                        if (error instanceof TimeoutError && warmupTries < 10) {
                            // note : may cause recursive invoke if always timeout.
                            textView.setText("Warming Up");
                            warmUp();
                            ++warmupTries;
                        } else {
                            textView.setText("Server Error");
                        }

                    }
                });

        mRequestQueue.add(stringRequest) ;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == Image_Capture_Code) {
            if (resultCode == RESULT_OK) {
                bp = (Bitmap) data.getExtras().get("data");
                imgCapture.setImageBitmap(bp);
                textView.setText("Sending");

                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                bp.compress(Bitmap.CompressFormat.JPEG, 100, stream);
                byte[] byteArray = stream.toByteArray();

                String encoded_str = Base64.encodeToString(byteArray, 0);

                HashMap<String, String> params = new HashMap<String, String>();
                params.put("image_bytes", encoded_str);

                Log.i("image", "image");
                JsonObjectRequest jsonObjectRequest = new JsonObjectRequest
                        (Request.Method.POST, url, new JSONObject(params), new Response.Listener<JSONObject>() {

                            @Override
                            public void onResponse(JSONObject response) {
                                textView.setText(response.toString());
                            }
                        }, new Response.ErrorListener() {

                            @Override
                            public void onErrorResponse(VolleyError error) {
                                textView.setText("Server Error, Try Again");
                            }
                        });

                mRequestQueue.add(jsonObjectRequest);

            } else if (resultCode == RESULT_CANCELED) {
                Toast.makeText(this, "Cancelled", Toast.LENGTH_LONG).show();
            }
        }
    }

}