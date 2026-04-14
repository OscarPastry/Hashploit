//CLI input → Load wordlist → Flatten data → Send to GPU →
//Run OpenCL kernel → Compare hashes → Return match
use ocl :: {Buffer, Context, Device, Kernel, Platform, Program, Queue, flags}; //ocl - The OpenCL                                                             
use clap :: Parser; //handles CLI arguments 
use std :: fs; // read files from disk 
use std::io::{BufRead, BufReader}; // for buffer reading
use std::time::Instant;
//CLI arguments
#[derive(Parser)]
#[command(name = "unhasher", about = "GPU-accelerated MD5 cracker")]
struct Args{
    #[arg(short = 'H', long)]
    hash: String,    // the MD5 hash you want to crack
    //./program -H abc123
    //OR
    //./program --Hash abc123
    #[arg(short, long)]
    wordlist: String, // path to your wordlist file 

    #[arg(short = 'D', long)]
    model : String,
}

struct Wordlist {
    words: Vec<String>, //original words ["cat", "dog", "hello"]
    flat: Vec<u8>, //all words as one big byte array [c,a,t,d,o,g,h,e,l,l,o]
    offsets: Vec<i32>, //where each word starts
    lengths: Vec<i32>, // how long is each word
}

fn load_wordlist_batch(lines_iter: &mut impl Iterator<Item = String>, batch_size: usize) -> Wordlist {
    let mut words = Vec::with_capacity(batch_size.min(1_000_000));
    let mut flat = Vec::with_capacity(batch_size.min(1_000_000) * 8);
    let mut offsets = Vec::with_capacity(batch_size.min(1_000_000));
    let mut lengths = Vec::with_capacity(batch_size.min(1_000_000));
    let mut offset = 0i32; // offset is init into a 32 bit int with value 0

    for line in lines_iter.take(batch_size) {
        let line = line.trim();
        if line.is_empty() || line.len() > 55 { continue; } // Removes empty lines , limits word lengths
        
        let bytes = line.as_bytes(); // converts &str -> & u8
        words.push(line.to_string()); //wsotres the words in the ram
        offsets.push(offset); // stores the current postion of word in flat vector
        lengths.push(bytes.len() as i32); // stores the length as a i32
        flat.extend_from_slice(bytes); // Appends all bytes of the current word to the flat
        offset += bytes.len() as i32; // Updates the offset for the next word
    }

    Wordlist {words, flat, offsets, lengths}
}

fn crack(target_hex: &str, wordlist_path: &str, gpu_model: &str) -> ocl :: Result <()>{//returns Ok(()) or Error
    // Parse target hash
    let target_bytes = hex::decode(target_hex)
        .expect("Invalid hex hash");
    assert_eq!(target_bytes.len(), 16, "Must be an MD5 hash(32 hex chars)");
    
    println!("[*] Target: {}", target_hex);
    
    //Pick Intel Arc GPU (or customized by param)
    //Lists all OpenCL platforms and picks the requested one
    let platform = Platform::list()
        .into_iter() // loop 
        .find(|p| { 
            p.name()
                .unwrap_or_default()
                .to_lowercase()
                .contains(gpu_model)
        })
        .unwrap_or_else(|| Platform::default()); //this is safeguard 
    
    let device = Device::first(platform)?; // ? operator :- If you find a device, give it to me; if you don't, 
                                           // return an error immediately and exit the function. 
    println!("[*] Using device: {}", device.name()?);

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?; // You can’t share memory buffers or kernels between different contexts. 
                  // Everything inside this context belongs to that specific hardware setup you identified in the previous steps.
    let queue = Queue::new(&context, device, None)?; // [None] refer to Queue Properties 
    
    //Load Kernel source
    let kernel_src = fs::read_to_string("kernels/crack.cl")
        .expect("Failed to read crack.cl");
    let program = Program::builder()
        .src(kernel_src)
        .build(&context)?;

    let target_buf = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(target_bytes.len())
        .copy_host_slice(&target_bytes)
        .build()?;
    
    let result_buf = Buffer::<i32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .len(1)
        .copy_host_slice(&[-1i32])
        .build()?;

    let file = fs::File::open(wordlist_path).expect("Failed to open wordlist");
    let reader = BufReader::new(file);
    let mut lines_iter = reader.lines().filter_map(|l| l.ok());

    let batch_size = 5_000_000;
    let mut batch_num = 1;
    let total_start = Instant::now();
    let mut total_words_processed = 0;

    loop {
        let wl = load_wordlist_batch(&mut lines_iter, batch_size);
        let num_words = wl.words.len();
        
        if num_words == 0 {
            println!("\n[-] NOT FOUND in wordlist. Total checked: {}", total_words_processed);
            break;
        }

        total_words_processed += num_words;
        println!("[*] Batch {}: launching {} GPU threads...", batch_num, num_words);
        let start = Instant::now();

        // allocate GPU buffers for this batch
        // We use max(1) to avoid allocating size 0 buffers which OpenCL dislikes
        let flat_buf = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
            .len(wl.flat.len().max(1))
            .copy_host_slice(&wl.flat)
            .build()?;

        let offsets_buf = Buffer::<i32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
            .len(wl.offsets.len().max(1))
            .copy_host_slice(&wl.offsets)
            .build()?;

        let lengths_buf = Buffer::<i32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
            .len(wl.lengths.len().max(1))
            .copy_host_slice(&wl.lengths)
            .build()?;

        //build and launch the kernel
        let kernel = Kernel::builder()
            .program(&program)
            .name("crack")
            .queue(queue.clone())
            .global_work_size(num_words)
            .arg(&flat_buf)
            .arg(&offsets_buf)
            .arg(&lengths_buf)
            .arg(&target_buf)
            .arg(&result_buf)
            .build()?;

        unsafe { kernel.enq()?; } 
        queue.finish()?;

        let elapsed = start.elapsed();
        let speed = num_words as f32 / elapsed.as_secs_f32() / 1_000_000.0;
        println!("    Speed: {:.2} MH/s | Time: {:.2?}", speed, elapsed);

        let mut result = vec![-1i32];
        result_buf.read(&mut result).enq()?;

        if result[0] != -1 {
            println!("\n[*] CRACKED: {}", wl.words[result[0] as usize]);
            println!("[*] Total Elapsed Time: {:.2?}", total_start.elapsed());
            break;
        }
        
        batch_num += 1;
    }

    Ok(())
}

fn main(){
    let args = Args::parse();
    crack(&args.hash, &args.wordlist, &args.model).expect("Crack failed");
}

