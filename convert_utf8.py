import codecs

try:
    with codecs.open("output_relearn_restored_50_9_clients.txt", "r", "utf-16le") as f_in:
        content = f_in.read()
    with codecs.open("output_utf8.txt", "w", "utf-8") as f_out:
        f_out.write(content)
    print("Conversion successful")
except Exception as e:
    print("Error:", e)
