using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CubeMovement : MonoBehaviour
{
    public float thrust;
    public Rigidbody rb;
    public Transform prefab;
    public float upwardSpeed;

    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    private void Update() {

    }

    void FixedUpdate(){

                // check to see if the user is pressing the Spacebar
        if(Input.GetKey(KeyCode.Space)){
            // once the Space key is pressed, apply force upwards onto gameObject (Player)
            rb.AddForce(new Vector3(0, upwardSpeed));
        }

        // moves cube to the left
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            Debug.Log("Left Arrow was pressed.");
            rb.AddForce(-thrust, 0, 0, ForceMode.Impulse);
        }

        // moves cube to the right
        if (Input.GetKey(KeyCode.RightArrow))
        {
            Debug.Log("Right Arrow was pressed.");
            rb.AddForce(thrust, 0, 0, ForceMode.Impulse);
        }

        // moves cube to the left
        if (Input.GetKey(KeyCode.A))
        {
            Debug.Log("A key was pressed.");
            rb.AddForce(-thrust, 0, 0, ForceMode.Impulse);
        }

        // moves cube to the right
        if (Input.GetKey(KeyCode.D))
        {
            Debug.Log("D key was pressed.");
            rb.AddForce(thrust, 0, 0, ForceMode.Impulse);
        }
    }
}
