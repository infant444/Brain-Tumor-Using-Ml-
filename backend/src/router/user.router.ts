import router from 'express';
import asynchandler from 'express-async-handler';
import bycrypt from 'bcryptjs';
import { User, UserModel } from '../model/user.model';
import Jwt from 'jsonwebtoken';
import nodeMailer from 'nodemailer';
import Mailgen from 'mailgen';
import { BAD_STATUS } from '../constant/status.constant';
import { Contect, ContectModel } from '../model/contect.model';
import { EMAILCOFIG } from '../config/email.config';
const rout = router()


rout.post("/signup", asynchandler(
    async (req, res) => {
        const { name, email, password, login_date, type } = req.body;
        const x = await UserModel.findOne({ 'email': email, 'type': type })
        console.log(x)
        if (x) {
            res.status(BAD_STATUS).send("Already exist")
            return;
        }
        const user: User = {
            name,
            email: email.toLowerCase(),
            profile: '',
            password: await bycrypt.hash(password, 10),
            type,
            login_date,
            admin: false
        }

        const userx = await UserModel.create(user)

        let transporter = nodeMailer.createTransport(EMAILCOFIG);
        let MailGenerator = new Mailgen({
            theme: "default",
            product: {
                name: '<h2 style=" width:100%;height:100%;background-color:#17252a;color:white;font: 2rem sans-serif;font-weight: 800;text-align: center; padding:2.5rem; margin:0">Neuro Scan</h2>',
                link: "http://localhost:4200/",
                logo:'../asset/logo.png',
            
            },
        })
        let response = {
            body: {
                name: name,
                intro: 'Welcome to Neuro Scan, your premier destination for cutting-edge neuroscience resources and solutions! We are thrilled to have you join our community of individuals passionate about the brain and nervous system.<br>At Neuro Scan, we are dedicated to providing you with the latest advancements, insights, and tools in the field of neuroscience. Whether you re a medical professional, researcher, student, or simply someone interested in learning more about the intricacies of the brain, we have something for you.',
                title: '<h2 style=" width:100%;color:#2b7a78;font: 2rem sans-serif;font-weight: 800;text-align: center;letter-spacing: 7px;">Welcome to Neuro Scan</h2>',
                outro: ' <p>If you have any questions or need assistance, please don\'t hesitate to contact us.</p><br><p>Best regards,<br>Neuro Scan Team</p>'
            }
        }

        let mail = MailGenerator.generate(response);
        let message = {
            from: '"🧠Neuro Scan" <' + process.env.EMAIL + '>',
            to: name + '<' + email + '>',
            subject: "Welcome to NeuroScan",
            html: mail,
        }

        transporter.sendMail(message).then(() => {

            res.send(genarateToken(userx))
            console.log("sucessfullay")
        }).catch(error => {
            res.status(BAD_STATUS).send(error);
        })
    }
))
rout.post("/login", asynchandler(
    async (req, res) => {
        const { email, password, type, login_date } = req.body;
        const user = await UserModel.findOne({ 'email': email, 'type': type })
        if (!user) {
            res.status(BAD_STATUS).send("Email is no exist")
            return;
        }
        if (await bycrypt.compare(password, user.password)) {
            const userx = await UserModel.findByIdAndUpdate(user.id, { 'login_date': login_date })
            console.log(user)
            let transporter = nodeMailer.createTransport(EMAILCOFIG);
            let MailGenerator = new Mailgen({
                theme: "default",
                product: {
                    name: '<h2 style=" width:100%;height:100%;background-color:#17252a;color:white;font: 2rem sans-serif;font-weight: 800;text-align: center; padding:2.5rem; margin:0">Neuro Scan</h2>',
                    link: "http://localhost:4200/",
                },
            })
            let response = {
                body: {
                    name: user.name,
                    intro: 'Welcome to Neuro Scan, your premier destination for cutting-edge neuroscience resources and solutions! We are thrilled to have you join our community of individuals passionate about the brain and nervous system.<br>At Neuro Scan, we are dedicated to providing you with the latest advancements, insights, and tools in the field of neuroscience. Whether you re a medical professional, researcher, student, or simply someone interested in learning more about the intricacies of the brain, we have something for you.',
                    title: '<h2 style=" width:100%;color:#2b7a78;font: 2rem sans-serif;font-weight: 800;text-align: center;letter-spacing: 7px;">Welcome to Neuro Scan</h2>',
                    outro: ' <p>If you have any questions or need assistance, please don\'t hesitate to contact us.</p><br><p>Best regards,<br>Neuro Scan Team</p>'
                }
            }

            let mail = MailGenerator.generate(response);
            let message = {
                from: '"🧠Neuro Scan" <' + process.env.EMAIL + '>',
                to: user.name + '<' + email + '>',
                subject: "Welcome to NeuroScan",
                html: mail,
            }

            transporter.sendMail(message).then(() => {

                res.send(genarateToken(userx))
                console.log("sucessfullay")
            }).catch(error => {
                res.status(BAD_STATUS).send(error);
            })
            res.send(genarateToken(userx))
            return;
        }
        res.status(BAD_STATUS).send("password is mismatch")
    }
))
rout.post("/login-google", asynchandler(
    async (req, res) => {
        const { name, email, profile, login_date, type } = req.body;
        const userx = await UserModel.findOne({ 'email': email, 'type': type })
        console.log(userx)
        if (userx) {
            await UserModel.findByIdAndUpdate(userx.id, { 'login_date': login_date })
            await UserModel.findByIdAndUpdate(userx.id, { 'profile': profile })
            const user=await UserModel.findById(userx.id)
            res.send(genarateToken(user))
            return;
        }
        const user: User = {
            name,
            email: email.toLowerCase(),
            profile,
            password: '',
            type,
            login_date,
            admin: false
        }

        const usery = await UserModel.create(user)
        let transporter = nodeMailer.createTransport(EMAILCOFIG);
        let MailGenerator = new Mailgen({
            theme: "default",
            product: {
                name: '<h2 style=" width:100%;height:100%;background-color:#17252a;color:white;font: 2rem sans-serif;font-weight: 800;text-align: center; padding:2.5rem; margin:0">Neuro Scan</h2>',
                link: "http://localhost:4200/",
            },
        })
        let response = {
            body: {
                name: user.name,
                intro: 'Welcome to Neuro Scan, your premier destination for cutting-edge neuroscience resources and solutions! We are thrilled to have you join our community of individuals passionate about the brain and nervous system.<br>At Neuro Scan, we are dedicated to providing you with the latest advancements, insights, and tools in the field of neuroscience. Whether you re a medical professional, researcher, student, or simply someone interested in learning more about the intricacies of the brain, we have something for you.',
                title: '<h2 style=" width:100%;color:#2b7a78;font: 2rem sans-serif;font-weight: 800;text-align: center;letter-spacing: 7px;">Welcome to Neuro Scan</h2>',
                outro: ' <p>If you have any questions or need assistance, please don\'t hesitate to contact us.</p><br><p>Best regards,<br>Neuro Scan Team</p>'
            }
        }

        let mail = MailGenerator.generate(response);
        let message = {
            from: '"🧠Neuro Scan" <' + process.env.EMAIL + '>',
            to: name + '<' + email + '>',
            subject: "Welcome to NeuroScan",
            html: mail,
        }

        transporter.sendMail(message).then(() => {
            console.log("sucessfullay")
            res.send(genarateToken(usery))
        }).catch(error => {
            res.status(BAD_STATUS).send(error);
        })
        return;
    }
))
const genarateToken = (user: any) => {
    const token = Jwt.sign({
        id: user.id,
        email: user.email,
        admin: user.admin
    }, process.env.USER_SECRATE_KEY!, {
        expiresIn: '48d',
    })
    return {
        id: user.id,
        name: user.name,
        email: user.email,
        profile: user.profile,
        type: user.type,
        token: token,
        login_date: user.login_date,
        admin: user.admin
    }
}

rout.post("/feedback", asynchandler(
    async (req, res) => {
        const { name, email, message, date } = req.body;
        const contect: Contect = {
            name,
            email,
            message,
            date,
            response:""
        }
        const contectx = await ContectModel.create(contect);
        res.send(contectx);
    }
))
rout.get("/send-otp/:email",asynchandler(
    async(req,res)=>{
        const user=await UserModel.findOne({ 'email': req.params.email, 'type': "email" })
        console.log(user)
        if(user){
            var otp="";
            for(var i=0;i<6;i++){
                otp=otp+Math.floor(Math.random()*10)as string
            }
            let transporter = nodeMailer.createTransport(EMAILCOFIG);
        let MailGenerator = new Mailgen({
            theme: "default",
            product: {
                name: 'Neuro Scan',
                link: "http://localhost:4200/",
            },
        })
        let response = {
            body: {
                name:"Dear :"+ user.name,
                intro: 'You recently requested to reset your password for your account. Please use the following One-Time Password (OTP) to reset your password:<br><h2>OTP: '+otp+'</h2>',
                title: 'Reset Password',
                outro: "Please note that this OTP is valid for a limited time and can only be used once. If you didn't request this change or if you have any questions, please contact our support team immediately.",
                signature:false
            }
        }

        let mail = MailGenerator.generate(response);
        let message = {
            from: '"🧠Neuro Scan" <' + process.env.EMAIL + '>',
            to: user.name+ '<' + user.email + '>',
            subject: "Reset Password",
            html: mail,
        }

        transporter.sendMail(message).then(() => {
            console.log("sucessfully")
            res.json({'otp':otp})
        }).catch(error => {
            res.status(BAD_STATUS).send(error);
        })
        }
        else{
            res.status(BAD_STATUS).send("email not exist");

        }
    }
))
rout.post("/changepassword/:email",asynchandler(
    async(req,res)=>{
        var {password}=req.body;
        password=password.toString()
        console.log(password)
        const user=await UserModel.findOne({ 'email': req.params.email, 'type': "email" })
        password=await bycrypt.hash(password,10);
        console.log(password)

        const userx=await UserModel.findByIdAndUpdate(user?.id,{password:password})
        res.send(userx);

    }
))
export default rout;