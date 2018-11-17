/*----------------------------------------------------------------------------*/
/**
 *	This confidential and proprietary software may be used only as
 *	authorised by a licensing agreement from ARM Limited
 *	(C) COPYRIGHT 2011-2012 ARM Limited
 *	ALL RIGHTS RESERVED
 *
 *	The entire notice above must be reproduced on all authorised
 *	copies and copies may only be made to the extent permitted
 *	by a licensing agreement from ARM Limited.
 *
 *	@brief	Functions for managing ASTC codec images.
 */
/*----------------------------------------------------------------------------*/

#include <math.h>

#include "astc_codec_internals.h"

#include "softfloat.h"
#include <stdint.h>
#include <stdio.h>


// fill the padding area of the input-file buffer with clamp-to-edge data
// Done inefficiently, in that it will overwrite all the interior data at least once;
// this is not considered a problem, since this makes up a very small part of total
// running time.

void fill_image_padding_area(astc_codec_image * img)
{
	if (img->padding == 0)
		return;

	int x, y, z, i;
	int exsize = img->xsize + 2 * img->padding;
	int eysize = img->ysize + 2 * img->padding;
	int ezsize = (img->zsize == 1) ? 1 : (img->zsize + 2 * img->padding);

	int xmin = img->padding;
	int ymin = img->padding;
	int zmin = (img->zsize == 1) ? 0 : img->padding;
	int xmax = img->xsize + img->padding - 1;
	int ymax = img->ysize + img->padding - 1;
	int zmax = (img->zsize == 1) ? 0 : img->zsize + img->padding - 1;


	// This is a very simple implementation. Possible optimizations include:
	// * Testing if texel is outside the edge.
	// * Looping over texels that we know are outside the edge.
	if (img->imagedata8)
	{
		for (z = 0; z < ezsize; z++)
		{
			int zc = MIN(MAX(z, zmin), zmax);
			for (y = 0; y < eysize; y++)
			{
				int yc = MIN(MAX(y, ymin), ymax);
				for (x = 0; x < exsize; x++)
				{
					int xc = MIN(MAX(x, xmin), xmax);
					for (i = 0; i < 4; i++)
						img->imagedata8[z][y][4 * x + i] = img->imagedata8[zc][yc][4 * xc + i];
				}
			}
		}
	}
	else if (img->imagedata16)
	{
		for (z = 0; z < ezsize; z++)
		{
			int zc = MIN(MAX(z, zmin), zmax);
			for (y = 0; y < eysize; y++)
			{
				int yc = MIN(MAX(y, ymin), ymax);
				for (x = 0; x < exsize; x++)
				{
					int xc = MIN(MAX(x, xmin), xmax);
					for (i = 0; i < 4; i++)
						img->imagedata16[z][y][4 * x + i] = img->imagedata16[zc][yc][4 * xc + i];
				}
			}
		}
	}
}




int determine_image_channels(const astc_codec_image * img)
{
	int x, y, z;

	int xsize = img->xsize;
	int ysize = img->ysize;
	int zsize = img->zsize;
	// scan through the image data
	// to determine how many color channels the image has.

	int lum_mask;
	int alpha_mask;
	int alpha_mask_ref;
	if (img->imagedata8)
	{
		alpha_mask_ref = 0xFF;
		alpha_mask = 0xFF;
		lum_mask = 0;
		for (z = 0; z < zsize; z++)
		{
			for (y = 0; y < ysize; y++)
			{
				for (x = 0; x < xsize; x++)
				{
					int r = img->imagedata8[z][y][4 * x];
					int g = img->imagedata8[z][y][4 * x + 1];
					int b = img->imagedata8[z][y][4 * x + 2];
					int a = img->imagedata8[z][y][4 * x + 3];
					lum_mask |= (r ^ g) | (r ^ b);
					alpha_mask &= a;
				}
			}
		}
	}
	else						// if( bitness == 16 )
	{
		alpha_mask_ref = 0xFFFF;
		alpha_mask = 0xFFFF;
		lum_mask = 0;
		for (z = 0; z < zsize; z++)
		{
			for (y = 0; y < ysize; y++)
			{
				for (x = 0; x < xsize; x++)
				{
					int r = img->imagedata16[z][y][4 * x];
					int g = img->imagedata16[z][y][4 * x + 1];
					int b = img->imagedata16[z][y][4 * x + 2];
					int a = img->imagedata16[z][y][4 * x + 3];
					lum_mask |= (r ^ g) | (r ^ b);
					alpha_mask &= (a ^ 0xC3FF);	// a ^ 0xC3FF returns FFFF if and only if the input is 1.0
				}
			}
		}
	}

	int image_channels = 1 + (lum_mask == 0 ? 0 : 2) + (alpha_mask == alpha_mask_ref ? 0 : 1);

	return image_channels;
}






// conversion functions between the LNS representation and the FP16 representation.

float float_to_lns(float p)
{

	if (astc_isnan(p) || p <= 1.0f / 67108864.0f)
	{
		// underflow or NaN value, return 0.
		// We count underflow if the input value is smaller than 2^-26.
		return 0;
	}

	if (fabs(p) >= 65536.0f)
	{
		// overflow, return a +INF value
		return 65535;
	}

	int expo;
	float normfrac = frexp(p, &expo);
	float p1;
	if (expo < -13)
	{
		// input number is smaller than 2^-14. In this case, multiply by 2^25.
		p1 = p * 33554432.0f;
		expo = 0;
	}
	else
	{
		expo += 14;
		p1 = (normfrac - 0.5f) * 4096.0f;
	}

	if (p1 < 384.0f)
		p1 *= 4.0f / 3.0f;
	else if (p1 <= 1408.0f)
		p1 += 128.0f;
	else
		p1 = (p1 + 512.0f) * (4.0f / 5.0f);

	p1 += expo * 2048.0f;
	return p1 + 1.0f;
}



uint16_t lns_to_sf16(uint16_t p)
{

	uint16_t mc = p & 0x7FF;
	uint16_t ec = p >> 11;
	uint16_t mt;
	if (mc < 512)
		mt = 3 * mc;
	else if (mc < 1536)
		mt = 4 * mc - 512;
	else
		mt = 5 * mc - 2048;

	uint16_t res = (ec << 10) | (mt >> 3);
	if (res >= 0x7BFF)
		res = 0x7BFF;
	return res;
}


// conversion function from 16-bit LDR value to FP16.
// note: for LDR interpolation, it is impossible to get a denormal result;
// this simplifies the conversion.
// FALSE; we can receive a very small UNORM16 through the constant-block.
uint16_t unorm16_to_sf16(uint16_t p)
{
	if (p == 0xFFFF)
		return 0x3C00;			// value of 1.0 .
	if (p < 4)
		return p << 8;

	int lz = clz32(p) - 16;
	p <<= (lz + 1);
	p >>= 6;
	p |= (14 - lz) << 10;
	return p;
}





void imageblock_initialize_deriv_from_work_and_orig(imageblock * pb, int pixelcount)
{
	int i;

	const float *fptr = pb->orig_data;
	const float *wptr = pb->work_data;
	float *dptr = pb->deriv_data;

	for (i = 0; i < pixelcount; i++)
	{

		// compute derivatives for RGB first
		if (pb->rgb_lns[i])
		{
			float r = MAX(fptr[0], 6e-5f);
			float g = MAX(fptr[1], 6e-5f);
			float b = MAX(fptr[2], 6e-5f);

			float rderiv = (float_to_lns(r * 1.05f) - float_to_lns(r)) / (r * 0.05f);
			float gderiv = (float_to_lns(g * 1.05f) - float_to_lns(g)) / (g * 0.05f);
			float bderiv = (float_to_lns(b * 1.05f) - float_to_lns(b)) / (b * 0.05f);

			// the derivative may not actually take values smaller than 1/32 or larger than 2^25;
			// if it does, we clamp it.
			if (rderiv < (1.0f / 32.0f))
				rderiv = (1.0f / 32.0f);
			else if (rderiv > 33554432.0f)
				rderiv = 33554432.0f;

			if (gderiv < (1.0f / 32.0f))
				gderiv = (1.0f / 32.0f);
			else if (gderiv > 33554432.0f)
				gderiv = 33554432.0f;

			if (bderiv < (1.0f / 32.0f))
				bderiv = (1.0f / 32.0f);
			else if (bderiv > 33554432.0f)
				bderiv = 33554432.0f;

			dptr[0] = rderiv;
			dptr[1] = gderiv;
			dptr[2] = bderiv;
		}
		else
		{
			dptr[0] = 65535.0f;
			dptr[1] = 65535.0f;
			dptr[2] = 65535.0f;
		}


		// then compute derivatives for Alpha
		if (pb->alpha_lns[i])
		{
			float a = MAX(fptr[3], 6e-5f);
			float aderiv = (float_to_lns(a * 1.05f) - float_to_lns(a)) / (a * 0.05f);
			// the derivative may not actually take values smaller than 1/32 or larger than 2^25;
			// if it does, we clamp it.
			if (aderiv < (1.0f / 32.0f))
				aderiv = (1.0f / 32.0f);
			else if (aderiv > 33554432.0f)
				aderiv = 33554432.0f;

			dptr[3] = aderiv;
		}
		else
		{
			dptr[3] = 65535.0f;
		}

		fptr += 4;
		wptr += 4;
		dptr += 4;
	}
}




// helper function to initialize the work-data from the orig-data
void imageblock_initialize_work_from_orig(imageblock * pb, int pixelcount)
{
	int i;
	float *fptr = pb->orig_data;
	float *wptr = pb->work_data;

	for (i = 0; i < pixelcount; i++)
	{
		if (pb->rgb_lns[i])
		{
			wptr[0] = float_to_lns(fptr[0]);
			wptr[1] = float_to_lns(fptr[1]);
			wptr[2] = float_to_lns(fptr[2]);
		}
		else
		{
			wptr[0] = fptr[0] * 65535.0f;
			wptr[1] = fptr[1] * 65535.0f;
			wptr[2] = fptr[2] * 65535.0f;
		}

		if (pb->alpha_lns[i])
		{
			wptr[3] = float_to_lns(fptr[3]);
		}
		else
		{
			wptr[3] = fptr[3] * 65535.0f;
		}
		fptr += 4;
		wptr += 4;
	}

	imageblock_initialize_deriv_from_work_and_orig(pb, pixelcount);
}




// helper function to initialize the orig-data from the work-data
void imageblock_initialize_orig_from_work(imageblock * pb, int pixelcount)
{
	int i;
	float *fptr = pb->orig_data;
	float *wptr = pb->work_data;

	for (i = 0; i < pixelcount; i++)
	{
		if (pb->rgb_lns[i])
		{
			fptr[0] = sf16_to_float(lns_to_sf16((uint16_t) wptr[0]));
			fptr[1] = sf16_to_float(lns_to_sf16((uint16_t) wptr[1]));
			fptr[2] = sf16_to_float(lns_to_sf16((uint16_t) wptr[2]));
		}
		else
		{
			fptr[0] = sf16_to_float(unorm16_to_sf16((uint16_t) wptr[0]));
			fptr[1] = sf16_to_float(unorm16_to_sf16((uint16_t) wptr[1]));
			fptr[2] = sf16_to_float(unorm16_to_sf16((uint16_t) wptr[2]));
		}

		if (pb->alpha_lns[i])
		{
			fptr[3] = sf16_to_float(lns_to_sf16((uint16_t) wptr[3]));
		}
		else
		{
			fptr[3] = sf16_to_float(unorm16_to_sf16((uint16_t) wptr[3]));
		}

		fptr += 4;
		wptr += 4;
	}

	imageblock_initialize_deriv_from_work_and_orig(pb, pixelcount);
}



/*
   For an imageblock, update its flags.

   The updating is done based on work_data, not orig_data.
*/
void update_imageblock_flags(imageblock * pb, int xdim, int ydim, int zdim)
{
	int i;
	float red_min = 1e38f, red_max = -1e38f;
	float green_min = 1e38f, green_max = -1e38f;
	float blue_min = 1e38f, blue_max = -1e38f;
	float alpha_min = 1e38f, alpha_max = -1e38f;

	int texels_per_block = xdim * ydim * zdim;

	int grayscale = 1;

	for (i = 0; i < texels_per_block; i++)
	{
		float red = pb->work_data[4 * i];
		float green = pb->work_data[4 * i + 1];
		float blue = pb->work_data[4 * i + 2];
		float alpha = pb->work_data[4 * i + 3];
		if (red < red_min)
			red_min = red;
		if (red > red_max)
			red_max = red;
		if (green < green_min)
			green_min = green;
		if (green > green_max)
			green_max = green;
		if (blue < blue_min)
			blue_min = blue;
		if (blue > blue_max)
			blue_max = blue;
		if (alpha < alpha_min)
			alpha_min = alpha;
		if (alpha > alpha_max)
			alpha_max = alpha;

		if (grayscale == 1 && (red != green || red != blue))
			grayscale = 0;
	}

	pb->red_min = red_min;
	pb->red_max = red_max;
	pb->green_min = green_min;
	pb->green_max = green_max;
	pb->blue_min = blue_min;
	pb->blue_max = blue_max;
	pb->alpha_min = alpha_min;
	pb->alpha_max = alpha_max;
	pb->grayscale = grayscale;
}


// Helper functions for various error-metric calculations

double clampx(double p)
{
	if (astc_isnan(p) || p < 0.0f)
		p = 0.0f;
	else if (p > 65504.0f)
		p = 65504.0f;
	return p;
}

// logarithm-function, linearized from 2^-14.
double xlog2(double p)
{
	if (p >= 0.00006103515625)
		return log(p) * 1.44269504088896340735;	// log(x)/log(2)
	else
		return -15.44269504088896340735 + p * 23637.11554992477646609062;
}


// mPSNR tone-mapping operator
double mpsnr_operator(double v, int fstop)
{
	int64_t vl = 1LL << (fstop + 32);
	double vl2 = (double)vl * (1.0 / 4294967296.0);
	v *= vl2;
	v = pow(v, (1.0 / 2.2));
	v *= 255.0f;
	if (astc_isnan(v) || v < 0.0f)
		v = 0.0f;
	else if (v > 255.0f)
		v = 255.0f;
	return v;
}


double mpsnr_sumdiff(double v1, double v2, int low_fstop, int high_fstop)
{
	int i;
	double summa = 0.0;
	for (i = low_fstop; i <= high_fstop; i++)
	{
		double mv1 = mpsnr_operator(v1, i);
		double mv2 = mpsnr_operator(v2, i);
		double mdiff = mv1 - mv2;
		summa += mdiff * mdiff;
	}
	return summa;
}




// Compute PSNR and other error metrics between input and output image
void compute_error_metrics(int compute_hdr_error_metrics, int input_components, const astc_codec_image * img1, const astc_codec_image * img2, int low_fstop, int high_fstop, int psnrmode)
{
	int x, y, z;
	static int channelmasks[5] = { 0x00, 0x07, 0x0C, 0x07, 0x0F };
	int channelmask;

	channelmask = channelmasks[input_components];


	double4 errorsum = double4(0, 0, 0, 0);
	double4 alpha_scaled_errorsum = double4(0, 0, 0, 0);
	double4 log_errorsum = double4(0, 0, 0, 0);
	double4 mpsnr_errorsum = double4(0, 0, 0, 0);

	int xsize = MIN(img1->xsize, img2->xsize);
	int ysize = MIN(img1->ysize, img2->ysize);
	int zsize = MIN(img1->zsize, img2->zsize);

	if (img1->xsize != img2->xsize || img1->ysize != img2->ysize || img1->zsize != img2->zsize)
	{
		printf("Warning: comparing images of different size:\n"
			   "Image 1: %dx%dx%d\n" "Image 2: %dx%dx%d\n" "Only intersection region will be compared.\n", img1->xsize, img1->ysize, img1->zsize, img2->xsize, img2->ysize, img2->zsize);
	}

	if (compute_hdr_error_metrics)
	{
		printf("Computing error metrics ... ");
		fflush(stdout);
	}

	int img1pad = img1->padding;
	int img2pad = img2->padding;

	double rgb_peak = 0.0f;

	for (z = 0; z < zsize; z++)
		for (y = 0; y < ysize; y++)
		{
			int ze1 = (img1->zsize == 1) ? z : z + img1pad;
			int ze2 = (img2->zsize == 1) ? z : z + img2pad;

			int ye1 = y + img1pad;
			int ye2 = y + img2pad;

			for (x = 0; x < xsize; x++)
			{
				double4 input_color1;
				double4 input_color2;

				int xe1 = 4 * x + 4 * img1pad;
				int xe2 = 4 * x + 4 * img2pad;

				if (img1->imagedata8)
				{
					input_color1 =
						double4(img1->imagedata8[ze1][ye1][xe1] * (1.0f / 255.0f),
								img1->imagedata8[ze1][ye1][xe1 + 1] * (1.0f / 255.0f), img1->imagedata8[ze1][ye1][xe1 + 2] * (1.0f / 255.0f), img1->imagedata8[ze1][ye1][xe1 + 3] * (1.0f / 255.0f));
				}
				else
				{
					input_color1 =
						double4(clampx(sf16_to_float(img1->imagedata16[ze1][ye1][xe1])),
								clampx(sf16_to_float(img1->imagedata16[ze1][ye1][xe1 + 1])),
								clampx(sf16_to_float(img1->imagedata16[ze1][ye1][xe1 + 2])), clampx(sf16_to_float(img1->imagedata16[ze1][ye1][xe1 + 3])));
				}

				if (img2->imagedata8)
				{
					input_color2 =
						double4(img2->imagedata8[ze2][ye2][xe2] * (1.0f / 255.0f),
								img2->imagedata8[ze2][ye2][xe2 + 1] * (1.0f / 255.0f), img2->imagedata8[ze2][ye2][xe2 + 2] * (1.0f / 255.0f), img2->imagedata8[ze2][ye2][xe2 + 3] * (1.0f / 255.0f));
				}
				else
				{
					input_color2 =
						double4(clampx(sf16_to_float(img2->imagedata16[ze2][ye2][xe2])),
								clampx(sf16_to_float(img2->imagedata16[ze2][ye2][xe2 + 1])),
								clampx(sf16_to_float(img2->imagedata16[ze2][ye2][xe2 + 2])), clampx(sf16_to_float(img2->imagedata16[ze2][ye2][xe2 + 3])));
				}

				rgb_peak = MAX(MAX(input_color1.x, input_color1.y), MAX(input_color1.z, rgb_peak));

				double4 diffcolor = input_color1 - input_color2;
				errorsum = errorsum + diffcolor * diffcolor;

				double4 alpha_scaled_diffcolor = double4(diffcolor.xyz * input_color1.w, diffcolor.w);
				alpha_scaled_errorsum = alpha_scaled_errorsum + alpha_scaled_diffcolor * alpha_scaled_diffcolor;

				if (compute_hdr_error_metrics)
				{
					double4 log_input_color1 = double4(xlog2(input_color1.x),
													   xlog2(input_color1.y),
													   xlog2(input_color1.z),
													   xlog2(input_color1.w));

					double4 log_input_color2 = double4(xlog2(input_color2.x),
													   xlog2(input_color2.y),
													   xlog2(input_color2.z),
													   xlog2(input_color2.w));

					double4 log_diffcolor = log_input_color1 - log_input_color2;

					log_errorsum = log_errorsum + log_diffcolor * log_diffcolor;

					double4 mpsnr_error = double4(mpsnr_sumdiff(input_color1.x, input_color2.x, low_fstop, high_fstop),
												  mpsnr_sumdiff(input_color1.y, input_color2.y, low_fstop, high_fstop),
												  mpsnr_sumdiff(input_color1.z, input_color2.z, low_fstop, high_fstop),
												  mpsnr_sumdiff(input_color1.w, input_color2.w, low_fstop, high_fstop));
					mpsnr_errorsum = mpsnr_errorsum + mpsnr_error;
				}
			}
		}

	if (compute_hdr_error_metrics)
	{
		printf("done\n");
	}

	double pixels = xsize * ysize * zsize;

	double num = 0.0;
	double alpha_num = 0.0;
	double log_num = 0.0;
	double mpsnr_num = 0.0;
	double samples = 0.0;

	if (channelmask & 1)
	{
		num += errorsum.x;
		alpha_num += alpha_scaled_errorsum.x;
		log_num += log_errorsum.x;
		mpsnr_num += mpsnr_errorsum.x;
		samples += pixels;
	}
	if (channelmask & 2)
	{
		num += errorsum.y;
		alpha_num += alpha_scaled_errorsum.y;
		log_num += log_errorsum.y;
		mpsnr_num += mpsnr_errorsum.y;
		samples += pixels;
	}
	if (channelmask & 4)
	{
		num += errorsum.z;
		alpha_num += alpha_scaled_errorsum.z;
		log_num += log_errorsum.z;
		mpsnr_num += mpsnr_errorsum.z;
		samples += pixels;
	}
	if (channelmask & 8)
	{
		num += errorsum.w;
		alpha_num += alpha_scaled_errorsum.w;	/* log_num += log_errorsum.w; mpsnr_num += mpsnr_errorsum.w; */
		samples += pixels;
	}

	double denom = samples;
	double mpsnr_denom = pixels * 3.0 * (high_fstop - low_fstop + 1) * 255.0f * 255.0f;

	double psnr;
	if (num == 0)
		psnr = 999.0;
	else
		psnr = 10.0 * log10((double)denom / (double)num);

	double rgb_psnr = psnr;

	if(psnrmode == 1)
	{
		if (channelmask & 8)
		{
			printf("PSNR (LDR-RGBA): %.6lf dB\n", psnr);

			double alpha_psnr;
			if (alpha_num == 0)
				alpha_psnr = 999.0;
			else
				alpha_psnr = 10.0 * log10((double)denom / (double)alpha_num);
			printf("Alpha-Weighted PSNR: %.6lf dB\n", alpha_psnr);

			double rgb_num = errorsum.x + errorsum.y + errorsum.z;
			if (rgb_num == 0)
				rgb_psnr = 999.0;
			else
				rgb_psnr = 10.0 * log10((double)pixels * 3 / (double)rgb_num);
			printf("PSNR (LDR-RGB): %.6lf dB\n", rgb_psnr);
		}
		else
			printf("PSNR (LDR-RGB): %.6lf dB\n", psnr);


		if (compute_hdr_error_metrics)
		{
			printf("Color peak value: %f\n", rgb_peak);
			printf("PSNR (RGB normalized to peak): %f dB\n", rgb_psnr + 20.0 * log10(rgb_peak));

			double mpsnr;
			if (mpsnr_num == 0)
				mpsnr = 999.0;
			else
				mpsnr = 10.0 * log10((double)mpsnr_denom / (double)mpsnr_num);
			printf("mPSNR (RGB) [fstops: %+d to %+d] : %.6lf dB\n", low_fstop, high_fstop, mpsnr);

			double logrmse = sqrt((double)log_num / (double)pixels);
			printf("LogRMSE (RGB): %.6lf\n", logrmse);
		}
	}
}

/*
	Main image loader function.

	We have specialized loaders for DDS, KTX and HTGA; for other formats, we use stb_image.
	This image loader will choose one based on filename.
*/



int get_output_filename_enforced_bitness(const char *output_filename)
{
	if (output_filename == NULL)
		return -1;

	int filename_len = strlen(output_filename);
	const char *eptr = output_filename + filename_len - 5;

	if (eptr > output_filename && (strcmp(eptr, ".htga") == 0 || strcmp(eptr, ".HTGA") == 0))
	{
		return 16;
	}

	eptr = output_filename + filename_len - 4;
	if (eptr > output_filename && (strcmp(eptr, ".tga") == 0 || strcmp(eptr, ".TGA") == 0))
	{
		return 8;
	}
	if (eptr > output_filename && (strcmp(eptr, ".exr") == 0 || strcmp(eptr, ".EXR") == 0))
	{
		return 16;
	}

	// file formats that don't match any of the templates above are capable of accommodating
	// both 8-bit and 16-bit data (DDS, KTX)
	return -1;
}
