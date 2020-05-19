using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraControl : MonoBehaviour
{
    // GameObject reference to the Cube
    public GameObject cube;

    // Hold offset value
    private Vector3 offset;

    // Horizontal and Vertical speed of the camera
    public float speedH = 2.0f;
    public float speedV = 2.0f;
    
    private float yaw = 0.0f;
    private float pitch = 0.0f;
    
    // Start is called before the first frame update
    void Start()
    {
        offset = transform.position - cube.transform.position;
    }

    // Update is called once per frame
    void Update () {
        yaw += speedH * Input.GetAxis("Mouse X");
        pitch -= speedV * Input.GetAxis("Mouse Y");

        transform.eulerAngles = new Vector3(pitch, yaw, 0.0f);
    }

    // LateUpdate is called once per frame - garnanteed to run after all item have been processed in the Update method is run
    void LateUpdate()
    {
        transform.position = cube.transform.position + offset;      
    }
}
