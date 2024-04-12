import router from 'express';
import asynchandler from 'express-async-handler'
import multer from 'multer'
import axios from 'axios'
import pdfdocument, { text } from 'pdfkit';
import auth from '../middleware/auth.mid';
import nodeMailer from 'nodemailer';
import Mailgen from 'mailgen';
import { Patient, PatientModel, Tumor_Info } from '../model/patient.model';
import { EMAILCOFIG } from '../config/email.config';
import { BAD_STATUS } from '../constant/status.constant';
import { UserModel } from '../model/user.model';

const rout = router()
const storage1 = multer.memoryStorage();

const upload = multer({
    storage: storage1,
    limits: { fieldSize: 1024 * 1024 }
});
rout.use(auth);

rout.post("/test", upload.single("image"), asynchandler(
    async (req, res) => {
        const { name, dob, age, gender, symptoms,user } = req.body;
        //   console.log(name)
        //   console.log(symptoms)
        const symptomsx=symptoms.split(",");
        try {
            console.log("hello")
            // const x=imgx.buffer
            const image = req.file?.buffer.toString('base64')
            const data = {
                img: image
            }
            const response = await axios.post("http://127.0.0.1:8000/api/briantumor", data)
            const result = response['data'];
            console.log(result['segmantation'])
            const decodedImage = Buffer.from(result['segmantation'], 'base64');
            const x:Tumor_Info={
                segmantation:result['segmantation'] ,
                classify: result['classify'],
                size: result['size'],
                confidence_level:result['confidence_level'],
            }
            var brain_tumor=true;
            var stage="#ff0000";
            if(result['classify']=='notumor'){
                 brain_tumor=false;
                 stage="#00ff00";

            }
            else{
                 brain_tumor=true;
                stage="#ff0000";

                
            }
            const patient:Patient={
                name,
                dob,
                age,
                gender,
                symptoms:symptomsx,
                given_img: image as string,
                predicted_img: decodedImage.toString('base64'),
                brain_tumor:brain_tumor,
                tumor_imfo: x,
                user_id: user
            }
            const patientx=await PatientModel.create(patient) 
            let transposer=nodeMailer.createTransport(EMAILCOFIG);
            let MailGenerator = new Mailgen({
                theme: "default",
                product: {
                    name: '<h2 style=" width:100%;height:100%;background-color:#17252a;color:white;font: 2rem sans-serif;font-weight: 800;text-align: center; padding:2.5rem; margin:0">Neuro Scan</h2>',
                    link: "http://localhost:4200/",
                },
            })
            var date= new Date().toLocaleTimeString('en-US', {
                hour: 'numeric', minute: '2-digit', second: "2-digit", day: "2-digit", month: "2-digit", year: "2-digit", hour12: true
              }).toString()
              const userx=await UserModel.findById(user);
            let Response = {
                body: {
                    name: user.name,
                    intro: 'We are excited to introduce Nero Scan, an innovative medical imaging platform designed to revolutionize the detection of brain tumors using cutting-edge machine learning technology. Nero Scan represents a significant advancement in the field of medical imaging, offering unparalleled accuracy and efficiency in identifying potential abnormalities in brain scans.',
                    title: '<h2 style=" width:100%;color:#2b7a78;font: 2rem sans-serif;font-weight: 800;text-align: center;letter-spacing: 7px;">Welcome to Neuro Scan</h2>',
                    table: {
                        color:"#808080",
                        border:"2px solid black",
                        title:"Tumor Information",
                        data: [
                          {
                            title:'<b>name:</b>',
                            content:patientx.name,
                          },
                          {
                            title:'<b>Date Of Birth</b>',
                            content:patientx.dob,
                          },
                          {
                            title:'<b>Age</b>',
                            content:patientx.age,
                          },
                          {
                            title:'<b>Report date&time</b>',
                            content:date,
                          },
                          {
                            title:'<b>Type</b>',
                            content:'<p style:"color:'+stage+'">'+patientx.tumor_imfo.classify+'</p>',
                          },
                          {
                            title:'<b>Tumor Size</b>',
                            content:patientx.tumor_imfo.size+"cm",
                          },
                        ],
                       
                        columns: {
                            customAlignment: {
                                title: 'left',
                                content:"left",
                            }
                        }
                    },
                    image:"data:image/jpeg;base64,"+patientx.given_img,
                    action:[
                        {
                            instructions:"To Download the report as a pdf",
                            button:{
                                text:"Download",
                                link:"http://localhost:4200/braintumor/"+patientx.id,
                                color:"#3aafa9",
                            },
                        }
                    ],
                    outro: ' <p>If you have any questions or need assistance, please don\'t hesitate to contact us.</p><br><p>Best regards,<br>Neuro Scan Team</p>',
                    signature:false,
                }
            }
            let mail = MailGenerator.generate(Response);
            let message = {
                from: '"🧠Neuro Scan" <' + process.env.EMAIL + '>',
                to: name + '<' +userx?.email  + '>',
                subject: "Nero Scan: Your Advanced Medical Image Processing Report",
                html: mail,
            }
    
            transposer.sendMail(message).then(() => {
            res.send(patientx)
            console.log("sucessfullay")
            }).catch(error => {
                console.log(error);
                res.status(BAD_STATUS).send(error);
            })

        } catch (c) {
            res.status(BAD_STATUS).send(c);
        }
    }
))

rout.get("/report/:userid", asynchandler(
    async (req, res) => {
        const patient=await PatientModel.findById(req.params.userid);
        var date= new Date().toLocaleTimeString('en-US', {
            hour: 'numeric', minute: '2-digit', second: "2-digit", day: "2-digit", month: "2-digit", year: "2-digit", hour12: true
          }).toString()
          var discribe="";
          var treatment="";
          if(patient?.tumor_imfo.classify=='glioma'){
             discribe="Glioma brain tumors typically exhibit hyperintensity on T2-weighted images, indicating increased water content and cellular density. They often display avid enhancement with contrast, highlighting areas of increased vascularity and disrupted blood-brain barrier. These imaging features aid in diagnosis and treatment planning for this aggressive form of brain cancer."
             treatment="Treatment modalities for glioma brain tumors include surgery, aiming for maximal safe resection, followed by radiation therapy targeting residual tumor cells. Radiation therapy utilizes high-energy radiation beams to destroy cancer cells or inhibit their growth. It's often combined with chemotherapy to enhance efficacy, particularly for aggressive or recurrent gliomas.";
          }
          else if(patient?.tumor_imfo.classify=='meningioma'){
             discribe="Meningioma brain tumors typically exhibit hyperintensity on T2-weighted images, indicating increased water content. They also display avid enhancement with contrast due to their rich vascularity. These characteristics aid in their identification and differentiation from surrounding brain tissue, crucial for diagnosis and treatment planning."
             treatment="Meningioma brain tumors can be treated through various modalities including surgery, radiation therapy, and sometimes chemotherapy. Radiation therapy, such as stereotactic radiosurgery or conventional radiotherapy, is a common treatment option, particularly for tumors that are inoperable or recur after surgery, aiming to shrink or control tumor growth through targeted radiation beams.";
          }
          else if(patient?.tumor_imfo.classify=='pituitary'){
             discribe="Pituitary brain tumors typically present as hyperintense lesions on T2-weighted MRI, indicating high water content. They exhibit avid contrast enhancement due to disrupted blood-brain barrier. Associated findings include mass effect, such as compression of adjacent structures, and involvement of the sellar region. Prompt diagnosis and management are crucial for optimal patient outcomes."
             treatment="Treatment modalities for pituitary brain tumors include surgery, often via transsphenoidal approach, to remove the tumor mass. Radiation therapy, such as stereotactic radiosurgery or conventional radiotherapy, may be employed to target residual or recurrent tumors. Medications, like dopamine agonists or somatostatin analogs, are used to manage hormone-secreting tumors or control tumor growth.";
          }else{
             discribe=""
          }
        const datex=date.split(',');
        const pdf = new pdfdocument({ margin: 1 });
        res.setHeader('Content-Type', 'application/pdf');
       res.setHeader('Content-Disposition', 'attachment; filename="report.pdf"');
        // const stream = pdf.pipe(res);
       pdf.pipe(res);
        pdf.image("./asset/logo.png", 10, 5, {
            fit: [75, 75],
        })
        pdf.font("./asset/White_Storm.otf").fontSize(35).fillColor('#2b7a78').text("Neuro Scan", 85, 25, {
            link: 'http://localhost:4200',
            continued: true
        },)
        pdf.font("Times-Roman").fontSize(15).fillColor("black").text("\nDate: "+datex[0]+"\nTime:"+datex[1], (pdf.page.width - 150), 20)
        pdf.strokeColor('black');
        pdf.lineWidth(2);
        pdf.moveTo(0, 90);
        pdf.lineTo(pdf.page.width, 90);
        pdf.stroke();
        const symproms=patient?.symptoms.join(', ');
        pdf.font("Times-Bold").fontSize(25).fillColor("black").text("Brain Tumor Test Report", 0, 95, { align: "center" })
        pdf.font("Times-Roman").fontSize(20).fillColor("black").text("Report id: "+patient?.id+"\nName:"+patient?.name+"\nAge:"+patient?.age+" \nDOB:"+patient?.dob+"\nsex:"+patient?.gender, 10, 130)
        pdf.font("Times-Roman").fontSize(20).fillColor("black").text("Dr.Joe bot\n", (pdf.page.width - 150), 130)
        pdf.font("Times-Bold").fontSize(18).fillColor("black").text("Clinical History:", 25, 270)
        pdf.font("Times-Roman").fontSize(16).fillColor("black").text("The patient presented with '"+symproms+"' .No previous history of neurological disorders. ", 30)

        if(patient?.brain_tumor){
        pdf.font("Times-Bold").fontSize(18).fillColor("black").text("Imaging Findings:", 25)
        pdf.font("Times-Bold").fontSize(18).fillColor("black").text("Type="+patient?.tumor_imfo.classify, 25)
        pdf.font("Times-Roman").fontSize(16).fillColor("black").text("MRI (Magnetic Resonance Imaging) of the brain was performed, revealing a well-defined, contrast-enhancing mass lesion measuring approximately "+patient?.tumor_imfo.size+" cm in diameter in the  region of the brain. The lesion demonstrates"+discribe+".", 30)
        pdf.font("Times-Bold").fontSize(18).fillColor("black").text("Machine Learning Prediction", 25)


        pdf.image("data:image/jpeg;base64,"+patient?.given_img, {
            fit: [128, 128],
        })
        // pdf.moveTo(x:125)
        pdf.moveDown()

        pdf.image("data:image/jpeg;base64,"+patient?.predicted_img ,{
            fit: [128, 128],
        })
        pdf.moveDown()

        pdf.font("Times-Roman").fontSize(16).fillColor("black").text("");
        pdf.moveDown()
        pdf.moveDown()
        pdf.moveDown()
        pdf.moveDown()
        pdf.font("Times-Roman").fontSize(16).fillColor("black").text("A machine learning model trained on a dataset of"+patient?.tumor_imfo.classify+" cases predicted the likelihood of the tumor being a "+patient?.tumor_imfo.classify+" with "+patient?.tumor_imfo.confidence_level+"% confidence. The model also provided insights into the tumor's growth pattern, potential invasiveness, and prognosis based on its radiomic features and molecular profile.", 30)
        pdf.font("Times-Bold").fontSize(18).fillColor("black").text("Treatment Plan:", 25)
        pdf.font("Times-Roman").fontSize(16).fillColor("black").text("The patient will be referred to the neurosurgery department for consideration of surgical resection. Adjuvant treatment with "+treatment+" may be recommended based on the tumor's characteristics and molecular profile.", 30)
        pdf.font("Times-Bold").fontSize(18).fillColor("black").text("Conclusion:", 25)
        pdf.font("Times-Roman").fontSize(16).fillColor("black").text("In summary,the brain imaging study conducted on "+datex[0]+" .The patient has been diagnosed with a "+patient?.tumor_imfo.classify+" brain tumor based on clinical, imaging, pathological, and machine learning findings. A multidisciplinary approach involving neurosurgery, pathology, radiology, and machine learning expertise will be essential for optimizing the patient's management and outcomes.", 30)
        pdf.end();
        }
        else{
            pdf.font("Times-Bold").fontSize(18).fillColor("black").text("Imaging Findings:", 25)
        pdf.font("Times-Bold").fontSize(18).fillColor("black").text("Type="+patient?.tumor_imfo.classify, 25)
            pdf.font("Times-Roman").fontSize(16).fillColor("black").text("Magnetic Resonance Imaging (MRI) of the brain reveals no evidence of any mass lesions, enhancing or non-enhancing abnormalities, or structural abnormalities within the brain parenchyma. Ventricular size and configuration are within normal limits. No midline shift or mass effect is appreciated.", 30)
            pdf.font("Times-Bold").fontSize(18).fillColor("black").text("Machine Learning Prediction", 25)
            pdf.font("Times-Roman").fontSize(16).fillColor("black").text("A machine learning model trained on a dataset of "+patient?.tumor_imfo.classify+" cases predicted the has being a "+patient?.tumor_imfo.classify+" with "+patient?.tumor_imfo.confidence_level+"% confidence. The model also provided insights into the tumor's growth pattern, potential invasiveness, and prognosis based on its radiomic features and molecular profile.", 30)
             pdf.font("Times-Bold").fontSize(18).fillColor("black").text("Conclusion:", 25)
            pdf.font("Times-Roman").fontSize(16).fillColor("black").text("In summary, the brain imaging study conducted on "+datex[0]+" demonstrates no evidence of intracranial tumor or abnormality. Clinical correlation may be necessary for comprehensive evaluation and ongoing management.", 30)
            pdf.end();
    
        }

    }
))

rout.get("/getreport/:userid",asynchandler(
    async(req,res)=>{
        const reports=await PatientModel.find({user_id:req.params.userid})
        res.send(reports);
    }
))
rout.get("/getsinglereport/:reportid",asynchandler(
    async(req,res)=>{
        const report=await PatientModel.findById(req.params.reportid)
        res.send(report);
    }
))


export default rout;