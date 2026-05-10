using NeuralNet.Core;
using NeuralNet.Core.Serialization;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace NeuralNet.Interface
{
    public partial class MainWindow : Window
    {
        WriteableBitmap drawn_digit;
        NeuralNetwork   network;
        public MainWindow()
        {
            InitializeComponent();
            drawn_digit         = new WriteableBitmap(28, 28, 96, 96, PixelFormats.Bgra32, null);
            img_drawing.Source  = drawn_digit;

            try
            {
                network = Serialization.Deserialize();
            }
            catch(Exception e) 
            {
                lbl_error.Content = $" {e.Message} \nFehler beim Laden";
            }
        }

        private void img_drawing_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                DrawPixel(e);
            }
        }
        private void DrawPixel(MouseEventArgs e)
        {
            var positon    = e.GetPosition(img_drawing);
            int x          = (int)(positon.X * 28 / img_drawing.ActualWidth);
            int y          = (int)(positon.Y * 28 / img_drawing.ActualHeight);
            int pen_width  = 2;
            int pen_height = 2;

            byte[] color  = new byte[pen_width * pen_height * 4];
            for(int pixel = 0; pixel < color.Length; pixel += 4)
            {
                color[pixel]     = 0;
                color[pixel + 1] = 0;
                color[pixel + 2] = 0;
                color[pixel + 3] = 255;
            }

            if (x >= 0 && x < 28 && y >= 0 && y < 28)
            {
                drawn_digit.WritePixels(new Int32Rect(x, y, pen_width, pen_height), color, pen_width * 4, 0);
            }
        }

        private void btn_send_Click(object sender, RoutedEventArgs e)
        {
            
            double[] answer     =   network.Predict(GetImageValues());
            lbl_result.Content  =   $"0 : {answer[0]:0.00}\n" +
                                    $"1 : {answer[1]:0.00}\n" +
                                    $"2 : {answer[2]:0.00}\n" +
                                    $"3 : {answer[3]:0.00}\n" +
                                    $"4 : {answer[4]:0.00}\n" +
                                    $"5 : {answer[5]:0.00}\n" +
                                    $"6 : {answer[6]:0.00}\n" +
                                    $"7 : {answer[7]:0.00}\n" +
                                    $"8 : {answer[8]:0.00}\n"+
                                    $"9 : {answer[9]:0.00}";
        }

        private double[] GetImageValues()
        {
            int     width       = drawn_digit.PixelWidth;
            int     height      = drawn_digit.PixelHeight;
            int     stride      = width * 4;

            byte[]  pixels      = new byte[height * stride];
            drawn_digit.CopyPixels(pixels, stride, 0);

            double[] values = new double[width * height];

            for(int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    int     index   = y * stride + x * 4;
                    byte    p       = pixels[index + 3];

                    values[y * width + x] = 1.0 - ((double)p / 255);
                }
            }

            return values;
        }

        private void btn_clear_Click(object sender, RoutedEventArgs e)
        {
            drawn_digit         = new WriteableBitmap(28, 28, 96, 96, PixelFormats.Bgra32, null);
            img_drawing.Source  = drawn_digit;
            lbl_result.Content  = "";
        }
    }
}
