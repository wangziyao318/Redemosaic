import json


results_filename = "results.json"




if __name__ == "__main__":
    try:
        with open(results_filename, "r") as f:
            results = json.load(f)
    except OSError:
        raise
    except json.decoder.JSONDecodeError:
        print(f"\n{results_filename} corrupted.")
        raise

    if results == {}:
        print("empty results")
        exit
    
    psnrs = [results[img]["psnr"] for img in results]

    print(psnrs[0])

    

    # for psnr_i in psnrs:
    #     best_R = max(psnr_i["R"] + psnr_i["G"] + psnr_i["B"], key=(psnr_i["R"] + psnr_i["G"] + psnr_i["B"]).get)
    #     best_Y = max(psnr_i["Y"], key=psnr_i["Y"].get)
