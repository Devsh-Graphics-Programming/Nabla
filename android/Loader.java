package @PACKAGE_NAME@;
import android.os.Build;
import android.app.Activity;
import android.view.WindowManager;
import android.os.Environment;
import android.content.Intent;
import android.net.Uri;
public class Loader extends android.app.NativeActivity
{
    /* load our native library */
    static {
        System.loadLibrary("@NATIVE_LIB_NAME@");
    }

    @Override
    protected void onCreate(android.os.Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // if we're running on something older than Android M (6.0), return now
        // before requesting permissions as it's not supported
        if(Build.VERSION.SDK_INT < Build.VERSION_CODES.M)
            return;

        String[] permission = new String[2];
        if (Build.VERSION.SDK_INT < 30 /*Build.VERSION_CODES.R*/) {
            permission[0] = android.Manifest.permission.WRITE_EXTERNAL_STORAGE;
            permission[1] = android.Manifest.permission.READ_EXTERNAL_STORAGE;
			
            requestPermissions(permission, 2);

            // Request is asynchronous, so prevent connection to server until permissions granted.
            while(checkSelfPermission(permission[0])
                != android.content.pm.PackageManager.PERMISSION_GRANTED)
            {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    break;
                }
            }
			while(checkSelfPermission(permission[1])
                != android.content.pm.PackageManager.PERMISSION_GRANTED)
            {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    break;
                }
            }
        }
        else {
            permission[0] = "android.permission.MANAGE_EXTERNAL_STORAGE"; // android.Manifest.permission.MANAGE_EXTERNAL_STORAGE;

            try {
                java.lang.reflect.Method method = Environment.class.getMethod("isExternalStorageManager");

                Boolean result = (Boolean)method.invoke(null);

                // Popup a dialog if we haven't granted Android storage permissions.
                if (!result) {
                    Intent viewIntent = new Intent( "android.settings.MANAGE_ALL_FILES_ACCESS_PERMISSION"  /*android.provider.Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION*/);
                    startActivity(viewIntent);
                }
            } catch(Exception e) { }
        }
    }
	
	public void openFolder(String path)
	{
		Intent intent = new Intent(Intent.ACTION_VIEW);
		Uri uri = Uri.parse(path);
		intent.setDataAndType(uri, "resource/folder");
		if (intent.resolveActivityInfo(getPackageManager(), 0) != null)
		{
			startActivity(Intent.createChooser(intent, "Open folder"));
		}
	}
}
