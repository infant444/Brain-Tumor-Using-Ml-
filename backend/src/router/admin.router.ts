import router from 'express';
import asynchandler from 'express-async-handler'
import auth from '../middleware/auth.mid';
import { UserModel } from '../model/user.model';
import { ContectModel } from '../model/contect.model';
import { ChatRoomModel } from '../model/chat.model';
import  nodemailer from 'nodemailer';
import Mailgen from 'mailgen';
import { BAD_STATUS } from '../constant/status.constant';
import { EMAILCOFIG } from '../config/email.config';


const rout=router(); 
rout.use(auth);

rout.get("/alluser",asynchandler(
    async(req,res)=>{
        const users=await UserModel.find();
        res.send(users)
    }
))

rout.get("/getuser/:id",asynchandler(
    async(req,res)=>{
        const user=await UserModel.findById(req.params.id)
        res.send(user)
    }
))
rout.delete("/delete/user/:id",asynchandler(
    async(req,res)=>{
        const dele=await UserModel.findByIdAndDelete(req.params.id);
        await ChatRoomModel.deleteMany({userid:req.params.id})
        res.send(dele);
    }
))
rout.get("/getallmessage",asynchandler(
    async(req,res)=>{
        const message=await ContectModel.find();
        res.send(message)
    }
))

rout.post("/sendmessage/:id",asynchandler(
    async(req,res)=>{
        const {response}=req.body;
        var messageinfo=await ContectModel.findById(req.params.id);
         messageinfo=await ContectModel.findByIdAndUpdate(req.params.id,{response:response});
        let transporter = nodemailer.createTransport(EMAILCOFIG);
        let MailGenerator = new Mailgen({
            theme: "default",
            product: {
                name: '<p style="color:#17252a;font: 1rem">Neuro Scan</h2>',
                link: "http://localhost:4200/",
            },
        })
        let responsex = {
            body: {
                name:messageinfo?.name,
                intro: response as string,
                title: '<h2 style=" width:100%;color:#000;font: 1rem sans-serif;font-weight: 800;text-align: center;letter-spacing: 7px;">Welcome to Neuro Scan</h2>',
                outro: ' <p>If you have any questions or need assistance, please don\'t hesitate to contact us.</p><br><p>Best regards,<br>Neuro Scan Team</p>',
                signature:false
            }
        }

        let mail = MailGenerator.generate(responsex);
        let message = {
            from: '"🧠Neuro Scan" <' + process.env.EMAIL + '>',
            to: messageinfo?.name + '<' + messageinfo?.email + '>',
            subject: "NeuroScan support",
            html: mail,
        }

        transporter.sendMail(message).then(() => {
            res.send(messageinfo)
            console.log("sucessfullay")
        }).catch(error => {
            res.status(BAD_STATUS).send(error);
        })
        return;
    }
))

export default rout;