DATASET_PATHS = [


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/progan',     
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/progan',
        data_mode='wang2020',
        key='progan'
    ),

    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/cyclegan',   
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/cyclegan',
        data_mode='wang2020',
        key='cyclegan'
    ),

    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/biggan/',   # Imagenet 
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/biggan/',
        data_mode='wang2020',
        key='biggan'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/stylegan',    
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/stylegan',
        data_mode='wang2020',
        key='stylegan'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/gaugan',    # It is COCO 
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/gaugan',
        data_mode='wang2020',
        key='gaugan'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/stargan',  
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/stargan',
        data_mode='wang2020',
        key='stargan'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/deepfake',   
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/deepfake',
        data_mode='wang2020',
        key='deepfake'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/seeingdark',   
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/seeingdark',
        data_mode='wang2020',
        key='sitd'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/san',   
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/san',
        data_mode='wang2020',
        key='san'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/crn',   # Images from some video games
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/crn',
        data_mode='wang2020',
        key='crn'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/imle',   # Images from some video games
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/imle',
        data_mode='wang2020',
        key='imle'
    ),
    

    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/imagenet',
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/guided',
        data_mode='wang2020',
        key='guided'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/laion',
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/ldm_200',
        data_mode='wang2020',
        key='ldm_200'
    ),

    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/laion',
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/ldm_200_cfg',
        data_mode='wang2020',
        key='ldm_200_cfg'
    ),

    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/laion',
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/ldm_100',
        data_mode='wang2020',
        key='ldm_100'
     ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/laion',
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/glide_100_27',
        data_mode='wang2020',
        key='glide_100_27'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/laion',
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/glide_50_27',
        data_mode='wang2020',
        key='glide_50_27'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/laion',
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/glide_100_10',
        data_mode='wang2020',
        key='glide_100_10'
    ),


    dict(
        real_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/laion',
        fake_path='/gs/fs/tgh-25IDF/chenyang/imdl_models/UniversalFakeDetect/dataset/test/diffusion/diffusion_datasets/dalle',
        data_mode='wang2020',
        key='dalle'
    ),



]
