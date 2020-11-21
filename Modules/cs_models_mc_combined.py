import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, LeakyReLU,  \
                                    MaxPooling2D, concatenate, UpSampling2D,\
                                    Multiply, ZeroPadding2D, Cropping2D

  

def fft_layer(image):
    # get real and imaginary portions
    real = Lambda(lambda image: image[:, :, :, 0])(image)
    imag = Lambda(lambda image: image[:, :, :, 1])(image)

    image_complex = tf.complex(real, imag)  # Make complex-valued tensor
    kspace_complex = tf.signal.fft2d(image_complex)

    # expand channels to tensorflow/keras format
    real = tf.expand_dims(tf.math.real(kspace_complex), -1)
    imag = tf.expand_dims(tf.math.imag(kspace_complex), -1)
    kspace = tf.concat([real, imag], -1)
    return kspace


def fft_layer_mc(image):
    nchannels = 24
    for ii in range(0,nchannels,2):

        # get real and imaginary portions
        real = Lambda(lambda image: image[:, :, :, ii])(image)
        imag = Lambda(lambda image: image[:, :, :, ii+1])(image)

        image_complex = tf.complex(real, imag)  # Make complex-valued tensor
        kspace_complex = tf.signal.fft2d(image_complex)

        # expand channels to tensorflow/keras format
        real = tf.expand_dims(tf.math.real(kspace_complex), -1)
        imag = tf.expand_dims(tf.math.imag(kspace_complex), -1)
        if ii == 0: 
            kspace = tf.concat([real, imag], -1)
        else:
            kspace = tf.concat([kspace,real, imag], -1)
    return kspace


def ifft_layer(kspace_2channel):
    #get real and imaginary portions
    real = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,0])(kspace_2channel)
    imag = Lambda(lambda kspace_2channel : kspace_2channel[:,:,:,1])(kspace_2channel)

    kspace_complex = tf.complex(real,imag) # Make complex-valued tensor
    image_complex = tf.signal.ifft2d(kspace_complex)

    # expand channels to tensorflow/keras format
    real = tf.expand_dims(tf.math.real(image_complex),-1)
    imag = tf.expand_dims(tf.math.imag(image_complex),-1)
    # generate 2-channel representation of image domain
    image_complex_2channel = tf.concat([real, imag], -1)
    return image_complex_2channel

def cnn_block(cnn_input, depth, nf, kshape,channels):
    """
    :param cnn_input: Input layer to CNN block
    :param depth: Depth of CNN. Disregarding the final convolution block that goes back to
    2 channels
    :param nf: Number of filters of convolutional layers, except for the last
    :param kshape: Shape of the convolutional kernel
    :return: 2-channel, complex reconstruction
    """
    layers = [cnn_input]

    for ii in range(depth):
        # Add convolutional block
        layers.append(Conv2D(nf, kshape, padding='same')(layers[-1]))
        layers.append(LeakyReLU(alpha=0.1)(layers[-1]))
    final_conv = Conv2D(channels, (1, 1), activation='linear')(layers[-1])
    rec1 = Add()([final_conv,cnn_input])
    return rec1


def unet_block(unet_input, kshape=(3, 3),channels = 2):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: 2-channel, complex reconstruction
    """

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(unet_input)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, kshape, activation='relu', padding='same')(conv4)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, kshape, activation='relu', padding='same')(conv5)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(64, kshape, activation='relu', padding='same')(conv6)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(48, kshape, activation='relu', padding='same')(conv7)

    conv8 = Conv2D(channels, (1, 1), activation='linear')(conv7)
    out = Add()([conv8, unet_input])
    return out

def unet_block2(unet_input, kshape=(3, 3),channels = 2):
    """
    :param unet_input: Input layer
    :param kshape: Kernel size
    :return: 2-channel, complex reconstruction
    """

    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(unet_input)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(96, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(96, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(96, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(192, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(192, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(192, kshape, activation='relu', padding='same')(conv3)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(96, kshape, activation='relu', padding='same')(up1)
    conv4 = Conv2D(96, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(96, kshape, activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(48, kshape, activation='relu', padding='same')(up2)
    conv5 = Conv2D(48, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(48, kshape, activation='relu', padding='same')(conv5)

    conv6 = Conv2D(channels, (1, 1), activation='linear')(conv5)
    out = Add()([conv6, unet_input])
    return out

def DC_block(rec,mask,sampled_kspace,sensitivities,channels,kspace = False):
    """
    :param rec: Reconstructed data, can be k-space or image domain
    :param mask: undersampling mask
    :param sampled_kspace:
    :param kspace: Boolean, if true, the input is k-space, if false it is image domain
    :return: k-space after data consistency
    """

    if kspace:
        rec_kspace = rec
    else:
        rec_kspace = Lambda(fft_layer)(rec)
    decoupled_rec = decouple_k_space(rec_kspace, sensitivities)
    decoupled_sampled = decouple_k_space(sampled_kspace, sensitivities)
    rec_kspace_dc =  Multiply()([decoupled_rec,mask])
    rec_kspace_dc = Add()([rec_kspace_dc,decoupled_sampled])
    return combine_k_space(rec_kspace_dc, sensitivities)

def decouple_k_space(combined_k_space, S):
    tf.keras.backend.set_floatx('float64')
    combined_image = Lambda(ifft_layer)(combined_k_space)
    real = Lambda(lambda image: image[:, :, :, 0])(combined_image)
    imag = Lambda(lambda image: image[:, :, :, 1])(combined_image)
    image_complex = tf.complex(real, imag)  # Make complex-valued tensor

    rank_4_image = tf.expand_dims(image_complex, -1)
    repeated_image = tf.broadcast_to(rank_4_image, S.shape)
    decoupled_images = tf.math.multiply(repeated_image, S)

    # get real and imaginary portions
    real = tf.math.real(decoupled_images)
    imag = tf.math.imag(decoupled_images)

    for ii in range(S.shape[-1]):
        real_to_concat = tf.expand_dims(real[:,:,:, ii],-1)
        imag_to_concat = tf.expand_dims(imag[:,:,:, ii],-1)
        if ii == 0:
            image_complex_mc = tf.concat([real_to_concat, imag_to_concat], -1)
        else:
            image_complex_mc = tf.concat([image_complex_mc, real_to_concat, imag_to_concat], -1)

    return fft_layer_mc(image_complex_mc)


def combine_k_space(k_space, S):
    nchannels = S.shape[-1]
    for i in range(0,nchannels,2):
        ii = i * 2
        #get real and imaginary portions
        real = Lambda(lambda kspace : k_space[:,:,:,ii])(k_space)
        imag = Lambda(lambda k_space : k_space[:,:,:,ii+1])(k_space)

        kspace_complex = tf.complex(real,imag) # Make complex-valued tensor
        image_complex = tf.signal.ifft2d(kspace_complex)

        temp = tf.math.multiply(image_complex, S[:,:,:,i])

        # expand channels to tensorflow/keras format
        real = tf.expand_dims(tf.math.real(temp),-1)
        imag = tf.expand_dims(tf.math.imag(temp),-1)
        if ii == 0: 
            # generate 2-channel representation of image domain
            image_complex_2channel = tf.concat([real, imag], -1)
        else:
            combined = tf.concat([real, imag], -1)
            image_complex_2channel = tf.math.add(image_complex_2channel, combined)

    return fft_layer(image_complex_2channel)

def deep_cascade_unet(depth_str='ki', H=218, W=170, Hpad = 3, Wpad = 3, kshape=(3, 3), channels = 12):

    parts = 2

    inputs = Input(shape=(H,W,parts))
    mask = Input(shape=(H,W,parts))
    sensitivities = Input(shape=(H,W,channels))
    layers = [inputs]
    kspace_flag = True
    for ii in depth_str:
        
        if ii =='i':
            # Add IFFT
            layers.append(Lambda(ifft_layer)(layers[-1]))
            kspace_flag = False
        # Add CNN block
        layers.append(ZeroPadding2D(padding=(Hpad,Wpad))(layers[-1]))
        layers.append(unet_block(layers[-1], kshape,parts))
        layers.append(Cropping2D(cropping=(Hpad,Wpad))(layers[-1]))
        
        # Add DC block
        layers.append(DC_block(layers[-1],mask,inputs,sensitivities,channels,kspace=kspace_flag))
        kspace_flag = True
    out = Lambda(ifft_layer)(layers[-1])
    model = Model(inputs=[inputs,mask,sensitivities], outputs=out)
    return model